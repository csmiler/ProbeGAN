import json
s = "{'IS': None, 'FID': {'all': 7.633376562722049, 'per_class': [24.644379310908107, 13.798353682435732, 24.067773494041603, 25.36098216712668, 13.982039904225303, 22.880550016075745, 18.15616785845208, 14.837189377338404, 14.724971279405054, 62.24465389379094]}}"
s = s.replace(",", " ").replace("[", " ").replace("]", " ").split()
m = []
for ch in s:
    try:
        n = float(ch)
        m.append(n)
    except:
        pass
print(m)

print("".join(["& {:.2f} ".format(n) for n in m]))

