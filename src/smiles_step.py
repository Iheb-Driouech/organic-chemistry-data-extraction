import os
import time
import json
import re
import ast
import unicodedata

import pandas as pd
import pubchempy as pcp
from openai import OpenAI

def smart_split_chem_list(text: str):
    """
    Split une liste de composés de façon robuste.
    - Ne coupe pas à l'intérieur de (), [], {}.
    - Évite de couper sur les virgules de nomenclature: (2Z,2'Z), 2,2'-, N,N'-, 1,4-, etc.
    - Coupe sur ; aussi (souvent séparateur).
    """
    if text is None:
        return []
    s = str(text).strip()
    if not s or s.lower() in {"n/a", "na", "none"}:
        return []

    # normaliser unicode (prime, etc.)
    s = unicodedata.normalize("NFKC", s)

    out = []
    buf = []
    depth_paren = depth_brack = depth_brace = 0

    def flush():
        token = "".join(buf).strip()
        buf.clear()
        if token:
            out.append(token)

    i = 0
    while i < len(s):
        ch = s[i]

        if ch == "(":
            depth_paren += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            buf.append(ch)
            i += 1
            continue
        if ch == "[":
            depth_brack += 1
            buf.append(ch)
            i += 1
            continue
        if ch == "]":
            depth_brack = max(0, depth_brack - 1)
            buf.append(ch)
            i += 1
            continue
        if ch == "{":
            depth_brace += 1
            buf.append(ch)
            i += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            buf.append(ch)
            i += 1
            continue

        at_top = (depth_paren == 0 and depth_brack == 0 and depth_brace == 0)

        # séparateurs candidats
        if at_top and ch in {",", ";"}:
            prev = s[i - 1] if i > 0 else ""
            nxt = s[i + 1] if i + 1 < len(s) else ""

            # Heuristiques anti-casse-nomenclature
            # 1) ne pas couper si autour on voit chiffres/prime (= positions 2,2′ etc.)
            if (prev.isdigit() and (nxt.isdigit() or nxt in {"'", "′"})) or (prev in {"'", "′"} and nxt.isdigit()):
                buf.append(ch)
                i += 1
                continue

            # 2) ne pas couper si motif "N,N" / "O,O" / "S,S" etc.
            # ex: N,N′- ; si on est sur la virgule après un caractère lettre majuscule et avant même lettre
            if prev.isalpha() and nxt.isalpha() and prev.upper() == nxt.upper():
                buf.append(ch)
                i += 1
                continue

            # 3) si virgule suivie d'un espace + lettre => probablement séparateur de liste
            # sinon on garde (virgule interne)
            if ch == "," and not (i + 2 < len(s) and s[i + 1] == " " and (s[i + 2].isalpha() or s[i + 2].isdigit())):
                buf.append(ch)
                i += 1
                continue

            # ok -> on split
            flush()
            i += 1
            continue

        buf.append(ch)
        i += 1

    flush()

    # post-trim
    out = [t.strip(" ,;") for t in out if t.strip(" ,;")]
    return out


def pubchem(name):
    try:
        smi = pcp.get_compounds(name, 'name')[0].isomeric_smiles
    except Exception as e:
        return 'Not Found'
    if smi and str(smi).strip():
        return str(smi).strip()
    return 'Not Found'


def opsin(name):
    jar_path = os.path.join(os.path.dirname(__file__), "opsin-cli-2.8.0-jar-with-dependencies.jar")
    if not os.path.exists(jar_path):
        return "Not Found"

    with open('input_temp.txt', 'w') as f:
        f.writelines([name])
        f.close()
    os.system(f'java -jar "{jar_path}" -osmi input_temp.txt output_temp.txt')
    if not os.path.exists("output_temp.txt"):
        return "Not Found"
    with open('output_temp.txt', 'r') as f:
        smi = f.readline()
        f.close()
    smi = smi.strip()
    if smi:
        return smi
    return "Not Found"


def get_name_from_llama(name, model="gpt-4o-mini", ):
    prompt = """
    Please extract the compounds or elements in the following dialogues and tell me their chemical names. You should response the name in a json format like {'name' : 'chemical name'}. The key 'name' is the origin name in the input. The value 'chemical name' is the chemical name of the compound or element. You shouldn't guess the chemical name of the raw name, and you should answer according to the name entered as much as possible. If the name refers to a class of compounds, please give a compound belonging to that class in the value 'chemical name' as an alternative. For example, halogens are replaced by chlorides, and alkyl groups are replaced by ethyl groups. If it is a complex mixture such as petroleum ether or alcohol, the answer should be ' none '.If you are not sure whether your answer is correct, you should answer 'none' in 'chemical name'.

    example:
        input: 
            o- and, predominantly, p-tolunitrile
        answer:
            {
                "o-tolunitrile": "3-Cyanotoluene",
                "p-tolunitrile": "4-Cyanotoluene"
            }

    input: 

    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
       model=model,
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt + name}
    ],
)

    result = response.choices[0].message.content

    # 1) enlever les fences ```json ... ```
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.strip())

# 2) essayer JSON strict d'abord
    try:
        json_dic = json.loads(clean)
    except json.JSONDecodeError:
    # fallback: parfois le modèle renvoie du dict Python avec quotes simples
        json_dic = ast.literal_eval(clean)

    vals = list(json_dic.values())
    out = []
    for v in vals:
       if isinstance(v, list):
           out.extend(v)
       else:
           out.append(v)
    return out



def get_smiles_with_trace(name, model="gpt-4o-mini"):
    trace = {
        "Original": name,
        "Candidate_used": "",
        "SMILES": "",
        "Status": "NOT_FOUND",
        "Route": "",
        "PubChem_result": "",
        "OPSIN_result": "",
        "LLM_suggestions": "",
        "Notes": "",
    }

    # 1) PubChem sur original
    try:
        smi = pubchem(name)
        if smi != "Not Found":
            trace.update({
                "Candidate_used": name,
                "SMILES": smi,
                "Status": "OK",
                "Route": "PUBCHEM",
                "PubChem_result": "FOUND",
                "OPSIN_result": "SKIPPED",
                "LLM_suggestions": "SKIPPED",
            })
            return trace
        trace["PubChem_result"] = "NOT_FOUND"
    except Exception as e:
        trace["PubChem_result"] = f"ERROR: {e}"

    # 2) OPSIN sur original
    try:
        smi = opsin(name)
        if smi != "Not Found":
            trace.update({
                "Candidate_used": name,
                "SMILES": smi,
                "Status": "OK",
                "Route": "OPSIN",
                "OPSIN_result": "FOUND",
                "LLM_suggestions": "SKIPPED",
            })
            return trace
        trace["OPSIN_result"] = "NOT_FOUND"
    except Exception as e:
        trace["OPSIN_result"] = f"ERROR: {e}"

    # 3) LLM: proposer un ou plusieurs noms alternatifs
    suggestions = []
    try:
        suggestions = get_name_from_llama(name, model=model)  # doit renvoyer une LISTE
        trace["LLM_suggestions"] = json.dumps(list(suggestions), ensure_ascii=False)
    except Exception as e:
        trace["LLM_suggestions"] = f"ERROR: {e}"
        suggestions = []

    # 4) Tester PubChem/OPSIN sur suggestions (dans l’ordre)
    for cand in suggestions:
        if not cand or str(cand).strip().lower() in ("none", "n/a"):
            continue
        cand = str(cand).strip()

        # PubChem cand
        try:
            smi = pubchem(cand)
            if smi != "Not Found":
                trace.update({
    "Candidate_used": cand,
    "SMILES": smi,
    "Status": "OK",
    "Route": "LLM->PUBCHEM",
    "PubChem_result": "FOUND_ON_CANDIDATE",
})
                return trace
        except Exception:
            pass

        # OPSIN cand
        try:
            smi = opsin(cand)
            if smi != "Not Found":
                trace.update({
    "Candidate_used": cand,
    "SMILES": smi,
    "Status": "OK",
    "Route": "LLM->OPSIN",
    "OPSIN_result": "FOUND_ON_CANDIDATE",
})
                return trace
        except Exception:
            pass

    # 5) Notes utiles (heuristique simple)
    low = name.lower()
    if any(k in low for k in ["electrode", "substrate", "wire", "glass", "ito", "ptfe"]):
        trace["Notes"] = "Likely material/object, not a discrete molecule."
    if "poly" in low or low.startswith("p(") or "blend" in low:
        trace["Notes"] = "Likely polymer/mixture; SMILES may be undefined."
    return trace

def run_smiles_lookup(
    input_table_csv,
    output_smiles_csv,
    model="gpt-4o-mini"
):
    """
    Pipeline step 4:
    Read *_table.csv and generate smiles_lookup.csv
    """

    input_data = pd.read_csv(input_table_csv)

    def explode_column(df, col, role):
        series = df[col].fillna("").astype(str)
        items = []
        for cell in series:
            for name in smart_split_chem_list(cell):
                items.append((name, role))

        out = pd.DataFrame(items, columns=["Name", "Role"])

        out["Name"] = out["Name"].str.strip()
        out = out[out["Name"].ne("")]
        out = out[~out["Name"].str.lower().isin(["n/a", "na", "none"])]
        out = out[~out["Name"].str.fullmatch(r"\d+")]

        return out

    react = explode_column(input_data, "Reactants", "Reactant")
    prod  = explode_column(input_data, "Products",  "Product")

    names_df = pd.concat([react, prod], ignore_index=True) \
                 .drop_duplicates(subset=["Name", "Role"])

    rows = []
    for _, r in names_df.iterrows():
        t = get_smiles_with_trace(r["Name"], model=model)
        t["Role"] = r["Role"]
        rows.append(t)

    output_data = pd.DataFrame(rows)
    output_data.to_csv(output_smiles_csv, index=False)

    return output_smiles_csv


if __name__ == '__main__':

    start = time.perf_counter()

    run_smiles_lookup(
        input_table_csv="../outputs/input_test_table.csv",
        output_smiles_csv="../outputs/smiles_lookup.csv",
        model="gpt-4o-mini"
    )

    end = time.perf_counter()
    print('runningtime:' + str(end - start))

    