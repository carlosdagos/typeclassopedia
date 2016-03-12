import Data.Maybe

data Maximum a = Maximum { getMaximum :: Maybe a } deriving (Show)
data Minimum a = Minimum { getMinimum :: Maybe a } deriving (Show)

instance Ord a => Monoid (Maximum a) where
    mempty = Maximum Nothing

    Maximum Nothing `mappend` x = x
    x `mappend` Maximum Nothing = x

    -- We're only left with the case of two Maximum values
    -- that are 'Just' and not 'Nothing', so the 'fromJust' function
    -- will not throw any errors
    Maximum x `mappend` Maximum y | j > k     = Maximum x
                                  | otherwise = Maximum y
                                    where
                                      j = fromJust x
                                      k = fromJust y

instance Ord a => Monoid (Minimum a) where
    mempty = Minimum Nothing

    Minimum Nothing `mappend` x = x
    x `mappend` Minimum Nothing = x

    -- Again, we can apply the same logic as before
    Minimum x `mappend` Minimum y | j < k      = Minimum x
                                  | otherwise  = Minimum y
                                    where
                                      j = fromJust x
                                      k = fromJust y

-- I feel like I just duplicated a lot of code :/
-- Futhermore, I feel like I just unwrapped a data type and wrapped it again
-- so perhaps we can optimize with
--
-- n@(Minimum x) `mappend` m@(Minimum y) | ... = n
--                                       | ... = m


-- The functions then are

maximum' :: (Ord a, Foldable t) => t a -> Maybe a
maximum' = getMaximum . foldMap (Maximum . Just)

minimum' :: (Ord a, Foldable t) => t a -> Maybe a
minimum' = getMinimum . foldMap (Minimum . Just)
