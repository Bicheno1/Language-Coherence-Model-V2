"""
Small simulation of two ideas: (1) a task assistant for applications (request -> action, with abstention) and
(2) separate bases per topic, with a router that decides which one to ask. Phrases in English.
"""
import re, math, collections, random, statistics
A = {  # action -> (4 training phrases, 4 test phrases worded differently)
 'bold':      (["make this text bold","bold the selected words","turn the highlighted text into bold","put the title in bold"],
               ["I want the selection to be bold","can you embolden this sentence","set the heading in bold type","make those words stand out in bold"]),
 'insert_table':(["insert a table with three columns","add a table below this paragraph","create a 4 by 5 table here","put a table in the document"],
               ["I need a table with two rows","draw a grid of cells under the text","add a spreadsheet-style table","place a table at the cursor"]),
 'export_pdf':(["export this document as a pdf","save a copy as pdf","convert the file to pdf","give me a pdf version"],
               ["I would like a pdf of this report","turn the doc into a pdf file","download it as pdf","produce a pdf from the current document"]),
 'undo':      (["undo the last change","go back one step","revert what I just did","undo that"],
               ["I made a mistake, take it back","cancel my last edit","step back please","reverse the previous action"]),
 'font_size': (["make the font bigger","increase the text size","set the font size to 14","make the letters smaller"],
               ["the text is too small, enlarge it","change the size of the characters to twelve","shrink the font a bit","use a larger font for this paragraph"]),
 'page_break':(["insert a page break","start a new page here","add a page break after this section","move the rest to the next page"],
               ["I want the next chapter on a fresh page","break the page after this paragraph","put a page break below the title","separate this part onto a new page"]),
 'save':      (["save the document","save my work","store this file now","save changes"],
               ["please keep what I wrote","write the file to disk","don't lose my changes, save them","persist the current document"]),
 'crop':      (["crop the image to a square","cut off the edges of the picture","trim this photo","crop to the selected area"],
               ["remove the borders of this photo","I only want the center part of the image","cut the picture down to a smaller frame","trim away the background edges"]),
 'rotate':    (["rotate the picture 90 degrees","turn the image to the left","flip the photo upside down","rotate it clockwise"],
               ["the photo is sideways, turn it upright","spin the image a quarter turn","twist the picture around","orient the image correctly by rotating it"]),
 'brightness':(["make the photo brighter","increase the brightness","lighten the image","the picture is too dark, brighten it"],
               ["add some light to this image","make it less dark","raise the exposure a little","boost the luminosity of the photo"]),
 'blur':      (["blur the background","add a blur to the image","soften the photo with a blur","apply a gaussian blur"],
               ["make the picture look fuzzy","defocus the edges of the photo","smooth this image with some blur","apply a blurring effect"]),
 'resize':    (["resize the image to 800 pixels wide","make the picture smaller","scale the photo down by half","change the image dimensions"],
               ["shrink this picture so it fits the page","I need the photo to be 500 by 500","enlarge the image to double its size","adjust the width and height of the picture"])}
OOS = ["what is the capital of France","tell me a joke","book a flight to Madrid","how is the weather today","play some music","send an email to my boss",
       "who won the match last night","remind me to call mom","what time is it in Tokyo","order a pizza","how do I cook rice","solve this equation for x",
       "write me a poem about the sea","turn off the lights","what is the meaning of life","delete the whole document","print three copies",
       "share it with my team","change the color of the sky in the photo","open a new browser tab"]
STOP={"the","a","an","to","of","this","that","it","is","i","me","my","in","on","at","and","for","with","as","be","please","can","you","want","need","would","like","do","just","some","one","by","so","up","than","what","how","who","are"}
def toks(s):
    out=[]
    for w in re.findall(r"[a-z0-9']+",s.lower()):
        if w in STOP: continue
        for suf in ("ing","ed","es","s"):
            if w.endswith(suf) and len(w)-len(suf)>=3: w=w[:-len(suf)]; break
        out.append(w)
    return out
def build(k):
    docs={a:[t for p in v[0][:k] for t in toks(p)] for a,v in A.items()}
    df=collections.Counter(w for d in docs.values() for w in set(d)); N=len(docs)
    idf=lambda w: math.log(1+N/(df.get(w,0)+0.5))
    vec=lambda ws:{w:(1+math.log(c))*idf(w) for w,c in collections.Counter(ws).items() if w in df}
    cents={a:vec(d) for a,d in docs.items()}
    return vec,cents
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a.values())); nb=math.sqrt(sum(x*x for x in b.values()))
    return sum(x*b.get(w,0) for w,x in a.items())/(na*nb) if na and nb else 0.0
def route(text,vec,cents):
    q=vec(toks(text)); sc=sorted(((cos(q,c),a) for a,c in cents.items()),reverse=True)
    return sc[0][1],sc[0][0],(sc[0][0]-sc[1][0])
print("="*86+"\nA) TASK ASSISTANT: 12 actions (7 from a word processor, 5 from an image editor), 48 rewritten test requests, 20 out of scope\n"+"="*86)
print("(correct = right action; error = acted with another action; abstains = did nothing)")
for k in (1,2,3,4):
    vec,cents=build(k); rows=[(a,t,*route(t,vec,cents)) for a,v in A.items() for t in v[1]]; oos=[route(t,vec,cents) for t in OOS]
    print(f"\ntraining examples per action: {k}")
    print(f"   {'similarity threshold':>20s}{'correct':>10s}{'error':>8s}{'abstains':>13s}   | {'out of scope: acts by mistake':>36s}")
    for tau in (0.0,0.15,0.25,0.35,0.45):
        ok=sum(1 for a,t,p,s,m in rows if s>=tau and p==a); err=sum(1 for a,t,p,s,m in rows if s>=tau and p!=a); ab=len(rows)-ok-err
        fa=sum(1 for p,s,m in oos if s>=tau)
        print(f"   {tau:>20.2f}{100*ok/len(rows):9.0f}%{100*err/len(rows):7.0f}%{100*ab/len(rows):12.0f}%   | {100*fa/len(oos):35.0f}%")
vec,cents=build(4); rows=[(a,t,*route(t,vec,cents)) for a,v in A.items() for t in v[1]]
print("\nerrors with 4 examples and threshold 0.25 (examples):")
for a,t,p,s,m in rows:
    if s>=0.25 and p!=a: print(f"   {t!r:55s} real={a:12s} eligio={p} (sim {s:.2f})")
print("out of scope requests that still act at threshold 0.25:",[ (t,route(t,vec,cents)[0]) for t in OOS if route(t,vec,cents)[1]>=0.25])
print("\n"+"="*86+"\nB) SEPARATE BASES PER TOPIC with a router (illustration with short texts)\n"+"="*86)
T={'medicine':["the doctor prescribed an antibiotic to treat the infection","penicillin was discovered by Alexander Fleming in 1928","a vaccine trains the immune system to recognize a virus","high blood pressure increases the risk of a heart attack","the surgeon removed the inflamed appendix","insulin helps the body control the level of sugar in the blood","symptoms of the flu include fever, cough and muscle pain","antibodies are proteins that fight bacteria and viruses"],
   'history':["the Roman Empire fell in the year 476","the French Revolution began in 1789 with the storming of the Bastille","Napoleon was defeated at the battle of Waterloo","the printing press spread books across Europe in the fifteenth century","the Second World War ended in 1945","the Aztec empire was conquered by Hernan Cortes","the Industrial Revolution started in Britain in the eighteenth century","ancient Egypt built pyramids as tombs for the pharaohs"],
   'science':["water is made of hydrogen and oxygen atoms","light travels at about three hundred thousand kilometers per second","plants use photosynthesis to turn sunlight into energy","gravity pulls objects toward the center of the earth","the periodic table organizes the chemical elements by atomic number","an electron has a negative electric charge","the moon orbits the earth about once a month","dna carries the genetic information of living things"]}
Q={'medicine':["which drug is used against bacterial infections","what does a vaccine do to the body","why is high blood pressure dangerous","what is insulin for","how does the flu show up in a patient"],
   'history':["when did the Roman Empire end","what happened at the Bastille","who lost at Waterloo","why were the pyramids built","when did the Second World War finish"],
   'science':["what are the atoms in water","how fast does light move","how do plants get their energy","what keeps objects on the ground","what charge does an electron carry"]}
BOTH=[("when was penicillin discovered and in which year did the war end",["medicine","history"]),("how do vaccines work and what is dna",["medicine","science"]),("who conquered the Aztecs and how did plants get energy",["history","science"])]
OOS2=["what is the best pizza recipe","who will win the next election","how do I fix my bicycle"]
docs={t:[w for s in S for w in toks(s)] for t,S in T.items()}; df=collections.Counter(w for d in docs.values() for w in set(d)); N=3
idf=lambda w: math.log(1+N/(df.get(w,0)+0.5)); vec2=lambda ws:{w:(1+math.log(c))*idf(w) for w,c in collections.Counter(ws).items() if w in df}; cents2={t:vec2(d) for t,d in docs.items()}
def scores(q): v=vec2(toks(q)); return sorted(((cos(v,c),t) for t,c in cents2.items()),reverse=True)
n=ok=0
for t,qs in Q.items():
    for q in qs: n+=1; ok+= scores(q)[0][1]==t
print(f"una pregunta -> su base: {ok}/{n} = {100*ok/n:.0f}%")
for tau in (0.0,0.1,0.2): print(f"   off-topic questions that would be answered at threshold {tau:.1f}: {sum(1 for q in OOS2 if scores(q)[0][0]>=tau)}/{len(OOS2)}")
print("questions that need TWO bases (they refer to each other): the ones above 0.15 are used")
for q,want in BOTH:
    s=scores(q); got=[t for sc,t in s if sc>=0.15]; print(f"   {q!r}\n      esperado {want}  elegido {got}  ({', '.join(f'{t} {sc:.2f}' for sc,t in s)})")
