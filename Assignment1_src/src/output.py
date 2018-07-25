import os


def savefig(fig, filename):
    filename = os.path.join("img", filename)
    print("Creating %s..." % (filename))
    fig.savefig(filename, dpi=150, bbox_inches='tight')
