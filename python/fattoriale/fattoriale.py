#29-09-2025
#programa per il calcolo del fattoriale

def fattoriale(n): #definizione di funzione
    if n==0:
        return 1
    if n==1:
        return n
    else:
        return n*fattoriale(n-1)
    
if __name__=="__main__": #blocco eseguito solo se il file è eseguito come script principale (*.py)
    for n in range (10):
        print(f"{n}!{fattoriale(n)}") #stampa riga per riga con f-string concatenando n,!e il fattoriale
