import warnings
warnings.filterwarnings("ignore")
import json
import sys
import requests
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import snapshot_download
from Rag_Emb import chat_with_repo
from fuzzywuzzy import process

def repo_similarity(query: str, descr: str) -> float:
    """
    Compute the similarity between the query and the description,
    using PyTorch tensors and util.pytorch_cos_sim.
    """
    # Encode query and tags as PyTorch tensors
    query_emb = light_model_emb.encode(query, convert_to_tensor=True) 
    descr_embs  = light_model_emb.encode(descr,  convert_to_tensor=True)     

    # Compute cosine similarity
    cos_scores = util.pytorch_cos_sim(query_emb, descr_embs)

    # Return similarity scores
    return cos_scores

def search(threshold):
    global base_threshold
    global language

    while(1):
        language = input("Which language? (Default C): ")

        if language.lower() in (''):
                language = 'c'

        if language.lower() == "exit":
            sys.exit()

        query = input("What repo you search?: ")

        while(1):
            language = language.lower()

            if language not in repos:
                print(f"❌ Language '{language}' not found.")
                language = input("🔁 Enter a valid language or press Enter to reset: ").strip().lower()
                if language == "":
                    break  # user wants to quit
                continue

            repo_entries = []
            for key, libraries in repos[language].items():
                if isinstance(libraries, list):
                    repo_entries.extend(libraries)

            best_matches = []
            for repo in repo_entries:
                comp_sim = repo_similarity(query, repo['description'])
                if comp_sim >= threshold:
                    best_matches.append((comp_sim, repo))
                else:
                    lib_tags = " ".join(repo['internal_tags'])
                    tags_sim = repo_similarity(query, lib_tags)
                    if tags_sim >= threshold:
                        best_matches.append((tags_sim, repo))

            best_matches.sort(key=lambda x: x[0], reverse=True)

            if best_matches:
                print(f"\n🔎 Found {len(best_matches)} match(es) above threshold {threshold:.2f}:\n")
                for sim, repo in best_matches:
                    print(f"🎯 Hashname:    {repo['hashname']}  (score: {sim.item():.2f})")
                    print(f"   URL:         {repo['url']}")
                    print(f"   Description: {repo.get('description','<no description>')}")
                    print(f"   Tags:        {', '.join(repo['internal_tags'])}\n")
                return  # Done successfully
            else:
                print(f"\n❌ No matches found above threshold {threshold:.2f}.")
                new_threshold = input("⬇ Set a lower threshold (Enter to retry): ").strip()
                if new_threshold in (''):
                    break
                try:
                    n_thresh = float(new_threshold)
                    if 0 <= n_thresh <= 1:
                        print("✅ Updated threshold.")
                        threshold = base_threshold = n_thresh
                    else:
                        print("❌ Threshold must be between 0 and 1.")
                        continue
                except ValueError:
                    print("❌ Invalid input. Not a number.")
                    continue

            
def handle_user_input(parts):
    """
    parts    : list of two strings [command_input, repo_input]
    language : current language key, e.g. "c"
    repos    : your dict of repos loaded from JSON

    Returns (command, repo_name) on success, or None on failure.
    """
    # 1) must have exactly two parts
    if len(parts) != 2:
        print("❌ Invalid input. Please use: readme <repo> or learn <repo>")
        return None

    cmd_in, repo_in = parts
    cmd_in = cmd_in.lower()

    # 2) fuzzy-match the command
    valid_cmds = ["readme", "learn"]
    cmd_match, cmd_score = process.extractOne(cmd_in, valid_cmds)
    if cmd_score < 80:
        print(f"❌ Unknown command '{cmd_in}' (score {cmd_score}%). Use readme or learn.")
        return "",""

    # 3) gather all hashnames under that language
    repo_entries = []
    for libs in repos[language].values():
        if isinstance(libs, list):
            repo_entries.extend(libs)
    hashnames = [r["hashname"] for r in repo_entries]

    # 5) fuzzy-match the repo name
    repo_match, repo_score = process.extractOne(repo_in.lower(), hashnames)
    if repo_score >= 95:
        # confident
        return cmd_match, repo_match

    if 85 <= repo_score < 95:
        yn = input(f"🤔 Did you mean '{repo_match}' ({repo_score}%)? [Y/n]: ").strip().lower()
        if yn in ("", "y"):
            return cmd_match, repo_match
        else:
            print("❌ Aborted.")
            return "", ""

    # too low confidence
    print(f"❌ No good match for '{repo_in}' (best: '{repo_match}' at {repo_score}%).")
    return "", ""

def post_search_menu():
    
    while True:
        print("\n-------------------")
        print("So, what do you want to do next?")
        print('Type   "readme <Hashname_repo>"   to view the README.md')
        print('Type   "learn <Hashname_repo>"    to chat with the repository')
        print('Press  Enter                      to perform another search')
        print("---------------------")

        user_input = input(">> ").strip()

        if user_input == "":
            return "search_again"

        parts = user_input.split(maxsplit=1)
        if len(parts) != 2:
            print("❌ Invalid input. Please use: readme <repo> or learn <repo>")
            continue

        command, repo_name = handle_user_input(parts)
        command = command.lower()


        if command == "readme":
            handle_readme(repo_name, language)
            continue

        elif command == "learn":
            handle_learn(repo_name)
            continue

        else:
            print("❌ Unknown command. Try again.")


def find_repo_by_hashname(hashname, language):
    libs = repos.get(language.lower(), {}).get("libraries", [])
    for lib in libs:
        if lib["hashname"].lower() == hashname.lower():
            return lib
    return None

def handle_readme(hashname, language):
    repo = find_repo_by_hashname(hashname, language)
    if not repo:
        print(f"❌ Repository '{hashname}' not found under language '{language}'.")
        return

    github_url = repo["url"]
    if not github_url.startswith("https://github.com/"):
        print("❌ Only GitHub repos are supported for now.")
        return

    # Extract user and repo name
    try:
        parts = github_url.replace("https://github.com/", "").replace(".git", "").split("/")
        user, repo_name = parts[0], parts[1]
    except Exception:
        print("❌ Invalid GitHub URL format.")
        return

    branches = ["main", "master"]
    readme_candidates = ["README.md", "readme.md", "README", "readme"]

    for branch in branches:
        for filename in readme_candidates:
            raw_url = f"https://raw.githubusercontent.com/{user}/{repo_name}/{branch}/{filename}"
            response = requests.get(raw_url)

            if response.status_code == 200:
                print(f"\n###########📖 {filename} from {repo_name} ({branch} branch):###########\n" + "-" * 40 + "\n")
                print(response.text)
                print(f"\n###########📖 END {filename} from {repo_name} ({branch} branch):###########\n" + "-" * 40 + "\n")
                return

    print("❌ README file not found in 'main' or 'master' branch.")
    

def handle_learn(repo_hash):
    print(f"🤖 Starting chat with: {repo_hash}")
    lib = find_repo_by_hashname(repo_hash, language)
    if not lib:
        print(f"❌ Repository '{repo_hash}' not found under language '{language}'.")
        return

    github_url = lib["url"]
    if not github_url.startswith("https://github.com/"):
        print("❌ Only GitHub repos are supported for now.")
        return

    chat_with_repo(repo_hash,github_url, language,RAG_sim_score_threshold=0.1)


def main():

    while (1):
       
        search(base_threshold)

        post_search_menu()

    return

if __name__ == "__main__":

    print("\n📚 Search Tool")
    print("======================")
    print("-> Set language (C,C++,Python)")
    print("-> Search repos (a btree C implementation?)")
    print("-> Learn and have fun!")
    print("======================")
    print("\n...But first we have to do some check")
    
    # Load JSON database
    with open('repos.json', 'r', encoding='utf-8') as f:
        repos = json.load(f)

    base_threshold = 0.5
    language ="c"
    light_model_emb_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Load the pre-trained Sentence-Transformer model
    # Check if model is cached
    try:
        light_model_emb_dir = snapshot_download(repo_id=light_model_emb_name, local_files_only=True)
        print(f"✅ Embedding Model  already downloaded")
        #print(f"✅ Embedding Model  already downloaded: {light_model_emb_dir}")
    except Exception:
        print("❌ Embedding Model not found locally.\n")
        print("Attention! It will be downloaded into your Hugging Face cache!")
        while True:
            yn_input = input("Do you want to download it now? (y/N): ").strip().lower()
            if yn_input == "y":
                light_model_emb = SentenceTransformer(light_model_emb_name)
                break
            elif yn_input in ("n"):
                print("🚫 Skipping model download.")
                light_model_emb = None
                break
            else:
                print("Please answer with 'y' or 'n'.")
        
    print("\nLoading Sentence-Transformer model...")
    light_model_emb = SentenceTransformer(light_model_emb_name)  # Lightweight and effective
   
    main()
