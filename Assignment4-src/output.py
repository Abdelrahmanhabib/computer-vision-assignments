import os

def savefig(fig, filename):
    print("Saving figure %s..." %filename)
    if not os.path.exists("out"):
        os.mkdir("out")
    fig.savefig(os.path.join("out", filename), dpi=200, bbox_inches='tight')
