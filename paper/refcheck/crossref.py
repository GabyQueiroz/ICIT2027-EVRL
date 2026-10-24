"""Look up candidate references on Crossref and Semantic Scholar; save metadata + abstracts."""
import json, sys, time, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding="utf-8")

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "refcheck/1.0 (mailto:research@example.org)"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            time.sleep(3 * (i + 1)); err = e
    return {"error": str(err)}

out = {}
for q in sys.argv[1:]:
    cr = get("https://api.crossref.org/works?rows=1&query.bibliographic=" + urllib.parse.quote(q))
    it = cr.get("message", {}).get("items", [{}])[0] if "message" in cr else {}
    doi = it.get("DOI")
    rec = dict(query=q, doi=doi, title=(it.get("title") or [""])[0],
               authors=[f"{a.get('given','')} {a.get('family','')}" for a in it.get("author", [])],
               venue=(it.get("container-title") or [""])[0], year=(it.get("issued", {}).get("date-parts") or [[None]])[0][0])
    if doi:
        ss = get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract,year,venue,citationCount")
        rec["abstract"] = ss.get("abstract")
        rec["citations"] = ss.get("citationCount")
        time.sleep(1.2)
    out[q] = rec
    print(json.dumps({k: rec[k] for k in ["doi", "title", "year", "venue"]}, ensure_ascii=False))
    print("   authors:", ", ".join(rec["authors"][:8]))
json.dump(out, open("refs_%d.json" % int(time.time()), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
