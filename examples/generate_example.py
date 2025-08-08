from tonnetz3d import Tonnetz3D

if __name__ == "__main__":
    t = Tonnetz3D(radius=2, prefer_sharps=True)
    t.save_html("tonnetz_example.html", highlight=("C", "maj"))
    print("Wrote tonnetz_example.html")
