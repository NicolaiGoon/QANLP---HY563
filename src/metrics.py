class metrics:
    def exactMatch(pred,gold):
        score = 0
        for p,g in zip(pred,gold):
            if p == g: score ++ 1
        return score / len(gold)