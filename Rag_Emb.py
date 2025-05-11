# Revised chat_with_txt.py with proper handling of generated tokens:
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import snapshot_download
from collections import OrderedDict
from gitingest import ingest

def chat_with_repo(hashname,github_url, language,RAG_sim_score_threshold=0.2):

    DEBUG=0
    RAG_DEBUG=0
    FAKE_STREAM=1

    if DEBUG:
        print(f"Torch CUDA: {torch.cuda.is_available()}")
        #print(torch.cuda.is_available())

    Rag_emb_model_name="sentence-transformers/all-MiniLM-L6-v2"

    try:
        Rag_model_emb_dir = snapshot_download(repo_id=Rag_emb_model_name, local_files_only=True)
        print(f"✅ Embedded-for-RAG Model  already downloaded")
        #print(f"✅ Chatting Model  already downloaded: {model_emb_dir}")
    except Exception:
        print("❌ Embedded-for-RAG Model not found locally.\n")
        print("Attention! It will be downloaded into your Hugging Face cache!")
        while True:
            yn_input = input("Do you want to download it now? (y/N): ").strip().lower()
            if yn_input == "y":
                break
            elif yn_input in ("n"):
                print("🚫 Skipping model download, ending chat")
                return
            else:
                print("Please answer with 'y' or 'n'.")

    # Load Sentence Transformer model for semantic embeddings
    Rag_emb_model = SentenceTransformer(Rag_emb_model_name)

    chat_model_name = "Qwen/Qwen3-0.6B" # <-- full PyTorch model

    print("\n...But first we have to do some other check")

    # Load the pre-trained Sentence-Transformer model
    # Check if model is cached
    try:
        chat_model_dir = snapshot_download(repo_id=chat_model_name, local_files_only=True)
        print(f"✅ Chatting Model  already downloaded")
        #print(f"✅ Chatting Model  already downloaded: {model_emb_dir}")
    except Exception:
        print("❌ Chatting Model not found locally.\n")
        print("Attention! It will be downloaded into your Hugging Face cache!")
        while True:
            yn_input = input("Do you want to download it now? (y/N): ").strip().lower()
            if yn_input == "y":
                break
            elif yn_input in ("n"):
                print("🚫 Skipping model download, ending chat")
                return
            else:
                print("Please answer with 'y' or 'n'.")


    pipe = pipeline("text-generation", model=chat_model_name, torch_dtype=torch.bfloat16, device_map="auto")

    summary, tree, content = ingest(f"{github_url}")

    '''
    # Read the document
    with open("temp_ingest.txt", "r", encoding="utf-8") as f:
        document = f.read()
    ''' 
    document = tree+content
        
    def split_document(document: str, lines_per_chunk: int) -> list[dict]:
        """
        1) Identify sections by the three-line header exactly:
            48×"=", newline
            FILE: <name>, newline
            48×"=", newline
        2) Include these header lines at the start of each section's text.
        3) Within each section, emit chunks of at most `lines_per_chunk` lines.
        Returns a list of dicts with:
        - global_id
        - group_id       (which section)
        - group_chunk_id (which chunk within that section)
        - text           (the joined lines, including headers)
        """
        eq48 = "=" * 48
        lines = document.splitlines(keepends=True)

        # 1) locate sections with headers, including header lines
        sections = []  # each entry: (filename, list_of_lines)
        i = 0
        n = len(lines)

        # skip preamble if any (optional group 0)
        preamble = []
        while i < n:
            if (lines[i].rstrip("\r\n") == eq48
                and i+2 < n
                and lines[i+1].startswith("File:")
                and lines[i+2].rstrip("\r\n") == eq48):
                break
            preamble.append(lines[i])
            i += 1
        if preamble:
            sections.append({"group_id": 1, "lines": preamble})

        # 2) for each header, capture header + section body
        group_id = 1
        while i < n:
            # detect header
            if (lines[i].rstrip("\r\n") == eq48
                and i+2 < n
                and lines[i+1].startswith("File:")
                and lines[i+2].rstrip("\r\n") == eq48):

                group_id += 1
                # include header lines
                header_lines = [lines[i], lines[i+1], lines[i+2]]
                filename = lines[i+1].strip().split("File:",1)[1].strip()
                i += 3
                body = []
                # collect until next header
                while not (
                    i < n
                    and lines[i].rstrip("\r\n") == eq48
                    and i+2 < n
                    and lines[i+1].startswith("File:")
                    and lines[i+2].rstrip("\r\n") == eq48
                ):
                    body.append(lines[i])
                    i += 1
                    if i >= n:
                        break
                # combine header + body
                sections.append({"group_id": group_id, "lines": header_lines + body})
                continue
            i += 1

        # 3) chunk each section into line-based chunks
        out = []
        global_id = 0
        for sec in sections:
            sid = sec["group_id"]
            sec_lines = sec["lines"]
            # break into sub-chunks of lines_per_chunk
            for chunk_idx, start in enumerate(range(0, len(sec_lines), lines_per_chunk), start=1):
                chunk_lines = sec_lines[start:start+lines_per_chunk]
                if not any(l.strip() for l in chunk_lines):
                    continue
                global_id += 1
                out.append({
                    "global_id": global_id,
                    "group_id": sid,
                    "group_chunk_id": chunk_idx,
                    "text": "".join(chunk_lines)
                })
        return out

    # Database of chunks
    Chunks_DB = split_document(document, lines_per_chunk=15)

    from collections import defaultdict, OrderedDict

    def retrieve(query, top_n=3, n_pre=3, n_post=3):
        
        query_embedding = Rag_emb_model.encode(query[0], convert_to_tensor=True)
        snipp_embedding = Rag_emb_model.encode(query[1], convert_to_tensor=True)
        # 1) score all chunks above threshold
        sims = []
        for chunk in Chunks_DB:
            chunk_emb = Rag_emb_model.encode(chunk['text'], convert_to_tensor=True)
            score_query  = util.pytorch_cos_sim(query_embedding, chunk_emb)[0].item()
            score_snipp  = util.pytorch_cos_sim(snipp_embedding, chunk_emb)[0].item()
            snipp_weight = 0.2
            avg_score    = (score_query + snipp_weight*score_snipp)
            if RAG_DEBUG:
                print(f"score_query: {score_query}")
                print(f"avg_score: {avg_score}")
            #print(avg_score)
            if avg_score >= RAG_sim_score_threshold:
                sims.append((chunk['global_id'], avg_score))

        # nothing relevant
        if not sims:
            return []

        # 2) pick top_n by score
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:top_n]

        # 3) build look-ups
        id_to_chunk = {c['global_id']: c for c in Chunks_DB}
        group_to_chunks = defaultdict(list)
        for c in Chunks_DB:
            group_to_chunks[c['group_id']].append(c)
        # sort each group by chunk id
        for lst in group_to_chunks.values():
            lst.sort(key=lambda c: c['group_chunk_id'])

        # 4) assemble with pre & post context
        seen_ids = set()
        seen_groups = set()
        out = []

        for gid, _ in top:
            c = id_to_chunk[gid]
            g = c['group_id']
            '''
            global_id_local1 = 0
            for cdb in Chunks_DB:
                if cdb['group_id'] == g and cdb['group_chunk_id'] ==1:
                    global_id_local1 = cdb['global_id']
            '''
            
            chunk_list = group_to_chunks[g]
            # find position of this chunk in its group
            idx = next(i for i,ch in enumerate(chunk_list) if ch['global_id'] == gid)

            if g not in seen_groups:
                global_id_local1_generator = (
                    chunk_loc1 for chunk_loc1 in chunk_list if chunk_loc1.get('group_id') == g and chunk_loc1.get('group_chunk_id') == 1
                )
                global_id_local1 = next(global_id_local1_generator, None)
                if global_id_local1.get('global_id') not in seen_ids:
                    if g not in seen_groups:
                        out.append(global_id_local1)
                        seen_groups.add(g)
                        seen_ids.add(c['global_id'])

            if RAG_DEBUG:
                print(f"cdb: {global_id_local1}")

            '''
            # — ensure we also include the very first chunk of each retrieved group
            if global_id_local1 not in seen_ids:
                if g not in seen_groups:
                    out.append(chunk_list[0])
                    seen_groups.add(g)
                    seen_ids.add(c['global_id'])
            '''
            # include n_pre predecessors
            for j in range(idx - n_pre, idx):
                if j >= 0:
                    pc = chunk_list[j]
                    if pc['global_id'] not in seen_ids:
                        out.append(pc)
                        seen_ids.add(pc['global_id'])

            # include the chunk itself
            if c['global_id'] not in seen_ids:
                out.append(c)
                seen_ids.add(c['global_id'])

            # include n_post successors
            for j in range(idx + 1, idx + 1 + n_post):
                if j < len(chunk_list):
                    nc = chunk_list[j]
                    if nc['global_id'] not in seen_ids:
                        out.append(nc)
                        seen_ids.add(nc['global_id'])
            
            # end loop

        # final ordering: within each group, by group_chunk_id
        grouped = OrderedDict()
        for c in out:
            grouped.setdefault(c['group_id'], []).append(c)

        out_ordered = []
        for gid, chunk_list in grouped.items():
            chunk_list.sort(key=lambda x: x['group_chunk_id'])
            out_ordered.extend(chunk_list)

        return out_ordered



    print("Ask questions about the repo (type 'exit' to quit):\n")
    while True:

        input_query = input('Ask me a question: ')

        '''
        messages = [
            {
                "role": "system",
                "content": "You are an helpful assistant.",
            },
            {"role": "user", "content": f"""I have to do an embedding search with other models, but they fails to retrieve code definition inside the file.
              So i want that you help me to formulate the input query.
              Based on the following instruction/query: \"{input_query}\".
              Write a snippet {language} code if can help the embedding search models.
              Or else return nothing."""
             },
        ]
        '''
        messages = [
                {
                "role": "system",
                "content": """You are an assistant specialized in optimizing embedding-based code search. 
                Your input is always a JSON-like dictionary with two fields: “input_query” (a natural-language question) and “language” (the target programming language). 

                Your task:
                1. Decide whether adding a minimal code snippet (function signature, class/struct/type definition, constant, docstring) would help an embedding-search model locate the relevant code in a large codebase.
                2. If yes, output **only** that snippet wrapped in triple-backticks with the correct language tag.
                3. If no, output **exactly** an empty string (zero characters).

                Constraints:
                - Snippet must be as short as possible and directly relevant.
                - Do **not** output any explanation, JSON wrapper, commentary, or additional text.
                - Always produce either a snippet or nothing—never null, never whitespace.

                Examples:
                - Query: “what is the license of the project?” → (empty output)
                - Query: “how the btree is coded and defined?” with language C →  
                ```c
                struct btree {
                    int order;
                    struct btree *children[MAX_CHILDREN];
                };
                ```""",
                },
                {
                    "role": "user",
                    "content": f"""  
                    "input_query": "{input_query}",  
                    "language": "{language}" """  
                }
            ]

        prompt = pipe.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True,
                                                    enable_thinking=True
                                                    )

        #response_snippet = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.0, top_k=10, top_p=0.95)
        response_snippet = pipe(prompt, max_new_tokens=1000, do_sample=False)

        full_output_snippet = response_snippet[0]["generated_text"]
        reply_only_snippet = full_output_snippet.partition('<think>')[2].partition('</think>')[2].strip()

        # Utils for debugging 
        #print(f"\n FULL RAG_REFORMULATION: {full_output_snippet.strip()}\n")
        #print(f"\n RAG_REFORMULATION: {reply_only_snippet.strip()}\n")
        if input_query.lower() == "exit":
            break
        retrieved = retrieve([input_query,reply_only_snippet], top_n=2)

        if RAG_DEBUG:
            print('Retrieved knowledge:\n')
            for c in retrieved:
                print(f" - (global:{c['global_id']}  group:{c['group_id']}.'{c['group_chunk_id']}') {c['text']}")
            print('END Retrieved knowledge:')
            print("\n")
        if RAG_DEBUG:
            context_text = None
            print('Retrieved knowledge:\n')
            for c in retrieved:
                context_text = "\n".join(f"{c['text']}" for c in retrieved)
            if context_text is None: # adjust this
                break
            print(context_text)
            print('END Retrieved knowledge:')
            print("\n")

        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        context_text = "\n".join(f"{c['text']}" for c in retrieved)    
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent AI system that respond within context.",
            },
            {"role": "user", "content": f"""Using the following context: \"{context_text}\". Answer the following question OR perform the following instruction: {input_query}
             If the information does not directly answer the question, state that you cannot find the answer in the provided context."""
             },
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True,
                                                    enable_thinking=True
                                                    )

        #response = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0., top_k=10, top_p=0.95)
        response = pipe(prompt, max_new_tokens=1000, do_sample=False)

        full_output = response[0]["generated_text"]
        #reply_only = full_output[len(prompt):]
        reply_only = full_output.partition('<think>')[2].partition('</think>')[2].strip()

        # Display the model's response
        if FAKE_STREAM:
            print("")
            print("Assistant: \n")
            to_stream = reply_only.strip()
            text_chunk_size = 2  # characters per “mini-token”
            for i in range(0, len(to_stream), text_chunk_size):
                piece = to_stream[i : i + text_chunk_size]
                print(piece, end="", flush=True)
                time.sleep(0.05)        # adjust delay to taste
            print("")
        else:
            print("")
            print(f"Assistant: {reply_only.strip()}\n")

        print()
