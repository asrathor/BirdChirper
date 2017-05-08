from subprocess import call

def main():
    for i in range(1,156):
        filepath = "CG/Canada Goose "+str(i)+".wav" #92
        print(filepath)
        newfile = "CG/CG"+str(i)+".wav"
        call(["mv",filepath, newfile]);

main()
