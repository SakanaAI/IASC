import jsonlines
import re

RULE = re.compile(r'<p class="schg" id="[^ ]*">')
MAIN_GROUP = re.compile(r'<h2>\d+ ')
PARENS = re.compile(r"\([^\(\)]*\)")
NON_SEGMENTS = re.compile(r"^[A-Z][a-z\-A-Z/]+\.?$")


def extract_phonemes(line):
  def phoneme(p):
    if NON_SEGMENTS.match(p):
      return False
    # Other detritus
    if p in ["nasal", "Alv.-pal."]:
      return False
    return True

  line = line.replace("<tr>", "")
  line = line.replace("</tr>", "")
  line = line.replace("<td>", "")
  line = line.replace("</td>", "")
  line = line.replace("{", "")
  line = line.replace("}", "")
  # Remove parenthesized ones since not clear what to do with them:
  line = PARENS.sub("", line)
  phonemes = [p for p in line.split() if phoneme(p)]
  return phonemes


links = set()
parents = {}


with open("diachronica.html") as s:
  link = ""
  data = []
  rules = []
  collect_proto_phonemes = False
  proto_phonemes = []
  proto_group = ""
  for line in s:
    line = line.strip()
    if line.startswith("<h2>"):
      if (
          "Contents" in line or
          "Preface" in line or
          "Licensing and " in line or
          "Contact Information" in line or
          "Changelog" in line or
          "Key to Abbreviations" in line
      ):
        continue
      elif "Most-Wanted" in line:
        break
      if MAIN_GROUP.match(line):
        proto_group = MAIN_GROUP.sub("", line)
        proto_group = proto_group.replace("</h2>", "").strip()
        if proto_group == "Vowel Shifts":
          proto_group = ""
        collect_proto_phonemes = True
        proto_phonemes = []
      link = line.replace("<h2>", "").replace("</h2>", "")
      ## Inconsistency in naming
      rules = []
      source = ""
    elif collect_proto_phonemes and line.startswith("<tr><td>"):
      proto_phonemes.extend(extract_phonemes(line))
    elif line.startswith('<p class="schg"'):
      collect_proto_phonemes = False
      rule = RULE.sub("", line).strip()
      rules.append(rule)
    elif link and line.startswith("<p>"):
      source = line.replace("<p>", "").replace("<i>", "").replace("</i>", "")
    elif "</section>" in line:
      if link and rules:
        link = " ".join(link.split()[1:])  # Remove section #
        parent = ""
        daughter = ""
        try:
          parent, daughter = link.split(" to ")
          parent = parent.strip()
          daughter = daughter.strip()
          parents[daughter] = parent
        except ValueError:
          pass
        data.append(
          {
            "link": link,
            "rules": rules,
            "source": source,
            "proto_phonemes": sorted(list(set(proto_phonemes))),
            "proto_group": proto_group,
            "language": daughter,
            "parent": parent,
          }
        )
        links.add(link)
      link = ""
      source = ""
      rules = []
      collect_proto_phonemes = False


def construct_chain(daughter, chain=[]):
  if daughter not in parents:
    return chain
  parent = parents[daughter]
  chain.append(f"{parent} to {daughter}")
  return construct_chain(parent, chain)


for elt in data:
  chain = construct_chain(elt["language"], chain=[])
  elt["chain"] = chain[-1::-1]


# Finally clean up the "Proto-" prefix, which is used inconsistently

for elt in data:
  elt["link"] = elt["link"].replace("Proto-", "")
  elt["chain"] = [link.replace("Proto-", "") for link in elt["chain"]]
  elt["parent"] = elt["parent"].replace("Proto-", "")


with jsonlines.open("diachronica.jsonl", "w") as writer:
  writer.write_all(data)
