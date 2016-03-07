{-
 Won't be pattern matching since we can't know all
 the type constructors for every monad
-}
data M a = M a a
data N a = N a a

instance Functor M where
    fmap f x = undefined

instance Functor N where
    fmap f x = undefined

instance Applicative M where
    pure x = M x x
    x <*> y = undefined

instance Applicative N where
    pure x = N x x
    x <*> y = undefined

instance Monad M where
    return x = M x x
    x >>= y = undefined

instance Monad N where
    return x = N x x
    x >>= y = undefined

join :: (Monad m) => m (m a) -> m a
join = undefined

distrib :: N (M a) -> M (N a)
distrib = undefined

join' :: M (N (M (N a))) -> M (N a)
join' x = (join $ x >>= return . distrib) >>= return . join

{-
  Proof:

  x                                                 :: M (N (M (N a)))
  x >>= return . distrib                            :: M (M (N (N a)))
  join $ x >>= return . distrib                     :: M (N (N a))
  (join $ x >>= return . distrib) >>= return . join :: M (N a)
-}
