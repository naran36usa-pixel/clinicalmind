import re
from llama_index.core.schema import TextNode

def split_by_section_with_config(config, full_text):
    """
    Step 2 & 3: Run section-aware chunking and map metadata taxonomy.
    """
    valid_sections = config.get("valid_sections")
    if not valid_sections:
        raise KeyError(f"Configuration for '{config.get('drug_name')}' lacks 'valid_sections' validation definitions.")

    print(f"  [Chunk] Executing regex structure parsing...")
    
    # Identify numbered sections like "1 INDICATIONS AND USAGE"
    section_pattern = r'\n(\d{1,2}\s+[A-Z][A-Z\s,/]{5,})\n'
    split_content = re.split(section_pattern, full_text)

    sections_found = {}
    current_section = None

    for item in split_content:
        item_stripped = item.strip()
        if not item_stripped:
            continue

        if re.match(r'^\d{1,2}\s+[A-Z][A-Z\s,/]{5,}$', item_stripped):
            section_name = re.sub(r'^\d{1,2}\s+', '', item_stripped).strip()
            if section_name in valid_sections:
                current_section = section_name
                sections_found[current_section] = ""
            else:
                current_section = None
        else:
            if current_section:
                sections_found[current_section] += item_stripped + "\n\n"

    # Turn text paragraphs into structured LlamaIndex nodes
    nodes = []
    for section_name, section_text in sections_found.items():
        if not section_text.strip():
            continue

        paragraphs = section_text.strip().split("\n\n")
        for idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < 50:  # Noise gate
                continue

            # Merge config data structure directly into LlamaIndex metadata
            node = TextNode(
                text=paragraph,
                metadata={
                    "drug_name": config["drug_name"],
                    "manufacturer": config["manufacturer"],
                    "therapeutic_area": config["therapeutic_area"],
                    "document_type": config["document_type"],
                    "section_name": section_name,
                    "paragraph_index": idx
                }
            )
            node.excluded_llm_metadata_keys = ["paragraph_index"]
            nodes.append(node)

    print(f"  [Chunk] Complete. Generated {len(nodes)} governance nodes.")
    return nodes