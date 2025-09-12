import json
from app.services.llm import llm_service
from app.services.vectorstore import vectorstore_service

# MINDMAP_PROMPT = """
# Tu es un expert académique. Ton rôle est d'extraire les concepts principaux 
# et leurs relations à partir du texte fourni. 

# Format JSON strict attendu :
# {{
#   "nodes": [
#     {{"id": "1", "label": "ConceptA"}},
#     {{"id": "2", "label": "ConceptB"}}
#   ],
#   "edges": [
#     {{"source": "1", "target": "2", "relation": "dépend de"}}
#   ]
# }}

# Texte :
# {context}
# """

MINDMAP_PROMPT = """
Tu es un expert académique. Ton rôle est d'extraire les concepts principaux et leurs relations sémantiques à partir du texte fourni, en produisant un mindmap structuré, riche et interconnecté.

Instructions strictes :
- Analyse le texte avec précision. Identifie tous les concepts mentionnés ou implicitement impliqués.
- Génère entre 10 et 18 nœuds au minimum. Chaque nœud doit être une phrase concise mais complète (pas de simples mots).
- Crée **au moins 20 relations** entre les nœuds. Un même nœud peut avoir plusieurs entrées et sorties.
- Utilise des verbes précis pour les relations :  
  "implique", "dépend de", "est un type de", "cause", "est nécessaire pour", "contraste avec",  
  "influence", "permet", "résulte en", "est contenu dans", "repose sur", "est un sous-ensemble de",  
  "contredit", "complète", "exclut", "renforce", "est une conséquence de", etc.
- Ne crée aucun concept hors du texte. Si le texte est trop court, retourne un JSON minimal mais valide.
- Le JSON doit être parfaitement valide : aucune virgule traînante, pas de commentaires, pas de guillemets doubles mal échappés.
- Les ID doivent être des chaînes de caractères numériques ("1", "2", ...), sans répétition.

Format JSON strict attendu :
{{
  "nodes": [
    {{"id": "1", "label": "La photosynthèse transforme l'énergie lumineuse en énergie chimique"}},
    {{"id": "2", "label": "Les chloroplastes sont les organites où se déroule la photosynthèse"}},
    {{"id": "3", "label": "La chlorophylle absorbe principalement la lumière bleue et rouge"}},
    {{"id": "4", "label": "Le dioxyde de carbone (CO₂) est un réactif essentiel"}},
    {{"id": "5", "label": "L'eau (H₂O) est utilisée comme donneur d'électrons"}},
    {{"id": "6", "label": "L'oxygène (O₂) est un sous-produit de la photolyse de l'eau"}},
    {{"id": "7", "label": "L'énergie lumineuse est captée par les pigments photosynthétiques"}},
    {{"id": "8", "label": "Le glucose est le produit final de la phase sombre de la photosynthèse"}},
    {{"id": "9", "label": "L'ATP est synthétisé lors de la phase lumineuse"}},
    {{"id": "10", "label": "Le NADPH est un transporteur d'électrons utilisé dans le cycle de Calvin"}},
    {{"id": "11", "label": "La lumière solaire est la source d'énergie initiale"}},
    {{"id": "12", "label": "Les stomates permettent l'entrée du CO₂ dans la feuille"}},
    {{"id": "13", "label": "La respiration cellulaire utilise le glucose produit par la photosynthèse"}},
    {{"id": "14", "label": "La production d'oxygène augmente la concentration atmosphérique en O₂"}},
    {{"id": "15", "label": "Le manque de lumière réduit la vitesse de la photosynthèse"}}
  ],
  "edges": [
    {{"source": "1", "target": "2", "relation": "se déroule dans"}},
    {{"source": "1", "target": "3", "relation": "nécessite"}},
    {{"source": "1", "target": "4", "relation": "utilise"}},
    {{"source": "1", "target": "5", "relation": "utilise"}},
    {{"source": "1", "target": "6", "relation": "produit"}},
    {{"source": "1", "target": "7", "relation": "dépend de"}},
    {{"source": "1", "target": "8", "relation": "aboutit à"}},
    {{"source": "1", "target": "9", "relation": "génère"}},
    {{"source": "1", "target": "10", "relation": "génère"}},
    {{"source": "1", "target": "11", "relation": "est alimentée par"}},
    {{"source": "1", "target": "12", "relation": "requiert l'ouverture de"}},
    {{"source": "1", "target": "13", "relation": "fournit le substrat à"}},
    {{"source": "1", "target": "14", "relation": "contribue à"}},
    {{"source": "1", "target": "15", "relation": "est limitée par"}},
    
    {{"source": "2", "target": "3", "relation": "contient"}},
    {{"source": "2", "target": "7", "relation": "héberge les pigments qui captent"}},
    {{"source": "2", "target": "9", "relation": "est le site de synthèse de"}},
    {{"source": "2", "target": "10", "relation": "est le site de réduction de"}},
    
    {{"source": "3", "target": "7", "relation": "est un pigment qui capte"}},
    {{"source": "3", "target": "15", "relation": "devient inefficace sans"}},
    
    {{"source": "4", "target": "8", "relation": "est incorporé dans"}},
    {{"source": "4", "target": "12", "relation": "entre par"}},
    
    {{"source": "5", "target": "6", "relation": "est décomposée pour libérer"}},
    {{"source": "5", "target": "10", "relation": "fournit les électrons pour réduire"}},
    
    {{"source": "6", "target": "14", "relation": "est libéré sous forme de"}},
    
    {{"source": "7", "target": "9", "relation": "permet la synthèse de"}},
    {{"source": "7", "target": "10", "relation": "permet la réduction de"}},
    
    {{"source": "8", "target": "13", "relation": "est le substrat de"}},
    {{"source": "8", "target": "15", "relation": "n'est pas produit en quantité suffisante sans"}},
    
    {{"source": "9", "target": "10", "relation": "est utilisé conjointement avec"}},
    {{"source": "9", "target": "8", "relation": "fournit l'énergie pour la synthèse de"}},
    
    {{"source": "10", "target": "8", "relation": "fournit les électrons et le H⁺ pour la synthèse de"}},
    
    {{"source": "11", "target": "1", "relation": "est la source d'énergie primordiale de"}},
    {{"source": "11", "target": "7", "relation": "est absorbée par"}},
    {{"source": "11", "target": "15", "relation": "son absence entraîne la réduction de"}},
    
    {{"source": "12", "target": "4", "relation": "est la voie d'entrée du"}},
    {{"source": "12", "target": "15", "relation": "fermées en cas de stress hydrique, ce qui limite la photosynthèse"}}
  ]
}}

Texte :
{context}
"""

def generate_mindmap(query: str) -> dict:
    # Récupération des chunks liés à la requête
    docs = vectorstore_service.query(query, k=5)
    context = "\n".join([doc.page_content for doc in docs])

    # Génération JSON via LLM
    prompt = MINDMAP_PROMPT.format(context=context)
    raw_response = llm_service.generate(prompt)
    
    print("=== RAW RESPONSE ===")
    print(raw_response)
    print("====================")

    try:
        # First try to parse the raw response directly
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from the response
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                data = json.loads(raw_response[start:end])
            except json.JSONDecodeError:
                raise ValueError("Impossible de parser la mindmap générée: JSON invalide")
        else:
            raise ValueError("Aucun JSON valide trouvé dans la réponse du modèle")
    
    # Validate and format the response to match MindmapResponse model
    if not isinstance(data, dict):
        raise ValueError("La réponse du modèle doit être un objet JSON")
        
    # Ensure required fields exist and have correct types
    if 'nodes' not in data or not isinstance(data['nodes'], list):
        raise ValueError("La réponse doit contenir un tableau 'nodes'")
    if 'edges' not in data or not isinstance(data['edges'], list):
        raise ValueError("La réponse doit contenir un tableau 'edges'")
    
    # Ensure nodes have required fields
    for i, node in enumerate(data['nodes']):
        if not isinstance(node, dict):
            raise ValueError(f"Le nœud {i} n'est pas un objet valide")
        if 'id' not in node or not isinstance(node['id'], str):
            raise ValueError(f"Le nœud {i} doit avoir un 'id' de type chaîne")
        if 'label' not in node or not isinstance(node['label'], str):
            raise ValueError(f"Le nœud {i} doit avoir un 'label' de type chaîne")
    
    # Ensure edges have required fields
    for i, edge in enumerate(data['edges']):
        if not isinstance(edge, dict):
            raise ValueError(f"La relation {i} n'est pas un objet valide")
        if 'source' not in edge or not isinstance(edge['source'], str):
            raise ValueError(f"La relation {i} doit avoir un 'source' de type chaîne")
        if 'target' not in edge or not isinstance(edge['target'], str):
            raise ValueError(f"La relation {i} doit avoir un 'target' de type chaîne")
        if 'relation' not in edge or not isinstance(edge['relation'], str):
            raise ValueError(f"La relation {i} doit avoir un 'relation' de type chaîne")
    
    return data

