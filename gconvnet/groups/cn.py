class cn:
    """Cyclic group of order n."""
    def __init__(self,n):
        assert n >= 1
        
        self.size = n
        self.identity = 0
        self.gens = {1}
        
    def inv(self,x):
        return (self.size - x) % self.size

    def mul(self,x,y):
        return (x+y) % self.size
    
    def __str__(self):
        return 'Z/' + self.size + 'Z'