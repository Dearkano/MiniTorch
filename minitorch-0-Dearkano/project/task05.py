from project.datasets import Simple, Split, Xor

N = 100


def simple_classify(pt):
    return 0.0 if pt[0] > 0.5 else 1.0


Simple(N, vis=True).graph("initial", model=simple_classify)


def classify(pt):
    "Classify based on x position"
    if pt[0] > 0.8 or pt[0] < 0.2:
        return 1.0
    else:
        return 0.0


Split(N, vis=True).graph("initial", model=classify)


def xor_classify(pt):
    if pt[0] < 0.5 and pt[1] < 0.5 or pt[0] > 0.5 and pt[1] > 0.5:
        return 0.0
    else:
        return 1.0


Xor(N, vis=True).graph("initial", model=xor_classify)
