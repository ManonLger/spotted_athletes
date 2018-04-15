from lecture_chiffres.read_bibs import read_with_crnn

# Read a bib number
print("Using CRNN module")
path = 'samples/bib/bib1.png'
C = read_with_crnn(path)
c = C.predict()
print(c)
