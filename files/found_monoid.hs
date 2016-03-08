data Found a = Found { getFound :: Maybe a }

instance Monoid (Found a) where
  mempty = Found Nothing
  Found Nothing `mappend` x = x
  x `mappend` _ = x

find' :: Foldable t => (a -> Bool) -> t a -> Maybe a
find' p = getFound . foldMap (\x -> if p x then Found (Just x) else Found Nothing)
