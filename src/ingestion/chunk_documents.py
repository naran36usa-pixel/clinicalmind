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
    
    # Hardened regex: Matches headings at the beginning of a line with or without section numbers
    section_regex = re.compile(
        r'^\s*(\d{1,2}\s+)?([A-Z0-9][A-Z0-9\s,\/\-\&\.\(\)]+)\s*$', 
        re.MULTILINE | re.IGNORECASE
    )

    boundaries = []
    
    # Diagnostic counter
    total_raw_matches = 0

    for match in section_regex.finditer(full_text):
        total_raw_matches += 1
        # Capture group 2 yields the text body excluding the leading digits
        raw_section_title = match.group(2).strip().upper()
        
        # Semantic mapping normalization layer
        normalized_title = raw_section_title
        if "DOSAGE" in raw_section_title and "ADMINISTRATION" in raw_section_title:
            normalized_title = "DOSAGE AND ADMINISTRATION"
        elif "ADVERSE" in raw_section_title and "REACTIONS" in raw_section_title:
            normalized_title = "ADVERSE REACTIONS"
        elif "SPECIFIC" in raw_section_title and "POPULATIONS" in raw_section_title:
            normalized_title = "USE IN SPECIFIC POPULATIONS"
            
        # Match check against configuration values
        if normalized_title in valid_sections:
            boundaries.append({
                "start": match.start(),
                "end": match.end(),
                "section_name": normalized_title
            })
            print(f"   [Boundary Match] Identified target partition: '{normalized_title}'")

    print(f"  [Chunk Diagnostic] Raw Regex Matches: {total_raw_matches} | Valid Config Sections Mapped: {len(boundaries)}")

    # Handle structural fall-through case
    if not boundaries:
        print("  ⚠️ [Chunk Warning] Zero structural boundaries matched the config profile. Indexing failed.")
        return []

    # Step 2: Slice text stream based on identified boundaries
    sections_found = {}
    for i, boundary in enumerate(boundaries):
        start_pos = boundary["end"]
        end_pos = boundaries[i + 1]["start"] if i + 1 < len(boundaries) else len(full_text)
        
        section_content = full_text[start_pos:end_pos].strip()
        if section_content:
            if boundary["section_name"] in sections_found:
                sections_found[boundary["section_name"]] += "\n\n" + section_content
            else:
                sections_found[boundary["section_name"]] = section_content

    # Step 3: Map text slices to LlamaIndex nodes
    nodes = []
    for section_name, section_text in sections_found.items():
        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        print(f"   [Node Factory] Processing section '{section_name}' — found {len(paragraphs)} raw text blocks.")
        
        for idx, paragraph in enumerate(paragraphs):
            if len(paragraph) < 60 or any(ignore in paragraph.lower() for ignore in ["page b", "continued"]):
                continue

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

    print(f"  [Chunk] Complete. Generated {len(nodes)} high-integrity governance nodes.")
    return nodes