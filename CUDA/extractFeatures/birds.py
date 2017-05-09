from subprocess import call

def main():
    call(["./extractFeatures", "AC/AC", "100"])
    call(["./extractFeatures", "AK/AK", "91"])
    call(["./extractFeatures", "AYW/AYW", "100"])
    call(["./extractFeatures", "BJ/BJ", "150"])
    call(["./extractFeatures", "CG/CG", "155"])
    call(["./extractFeatures", "GC/GC", "102"])
    call(["./extractFeatures", "GWT/gwt", "63"])
    call(["./extractFeatures", "NC/NC", "57"])

main()
