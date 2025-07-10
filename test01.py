import random

a=int(input("adivinha o número em que eu pensei : "))
numero_secreto=random.randint(1, 10)
print("Testaste o numero  -> ",a)
print("O numero secreto é -> ", numero_secreto)
if numero_secreto==a:
    print("Acertaste!!")
else:
    print("Erraste. És uma grande besta !!!")



