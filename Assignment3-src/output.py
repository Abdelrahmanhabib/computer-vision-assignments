import os

def savefig(fig, filename):
    print("Saving figure %s..." %filename)
    fig.savefig(os.path.join("out", filename), dpi=200, bbox_inches='tight')
