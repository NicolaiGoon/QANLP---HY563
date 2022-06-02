class metrics:
    def exactMatch(pred,gold):
        '''
        exact match metric
        '''
        assert(len(pred) == len(gold))
        score = 0
        for p,g in zip(pred,gold):
            if p == g: score += 1
        return score / len(pred)
    
    def f1score(pred,gold):
        '''
        micro f1
        '''
        assert(len(pred) == len(gold))
        score = 0
        for p,g in zip(pred,gold):
            common = sum([1 if word in g.split(' ') else 0  for word in p.split(' ')])
            precision = common/len(p.split(' '))
            recall = common/len(g.split(' '))
            try:
                f1 = 2 * (precision * recall) / (precision + recall)
            except:
                f1 = 0
            score += f1
        return score / len(pred)