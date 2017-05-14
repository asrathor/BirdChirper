from subprocess import call

def main():
    for i in range(1,64):
        filepath = "GWT/gwt"+str(i)+".wav"
        print(filepath)
        newfile = "GWT/GWT"+str(i)+".wav"
        call(["mv",filepath, newfile]);

main()
