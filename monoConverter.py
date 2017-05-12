from pydub import AudioSegment

def monoconvert(filename, num):
    sound = AudioSegment.from_wav(filename)
    sound = sound.set_channels(1)
    sound.export("NC/NC"+str(num)+".wav", format="wav")

for i in range(1,58):
    #filepath = "American Kestrel/AK"+str(i)+".wav" #92
    #filepath = "American Yellow Warbler/AYW"+str(i)+".wav" #101
    #filepath = "Blue Jays/Blue Jay "+str(i)+".wav" #151
    #filepath = "Canada Goose/Canada Goose "+str(i)+".wav" #156
    filepath = "Northern Cardinal/Northern Cardinal "+str(i)+".wav" #58`
    monoconvert(filepath, i)
