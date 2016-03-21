data Free f a = Var a | Node (f (Free f a))

instance Functor (Free f) where
    fmap = undefined

instance (Functor f) => Applicative (Free f) where
    pure    = undefined
    _ <*> _ = undefined

instance (Functor f) => Monad (Free f) where
    return       = Var
    Var x  >>= h = h x
    Node x >>= h = Node (fmap (>>= h) x) 
