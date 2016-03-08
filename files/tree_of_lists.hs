{}
import Data.Tree

appendElement :: (Eq a) => a -> [a] -> [a]
appendElement a xs | a `elem` xs = xs
                   | otherwise   = a:xs

getLevel :: Int -> [[a]] -> [a]
getLevel n ls | n <= (length ls) - 1 = ls !! n
              | otherwise            = []

treeOfLists :: [Tree a] -> Tree [a]
treeOfLists xs = undefined 

firstTree :: Tree Int
firstTree = Node 1 [(Node 2 []), (Node 3 []), (Node 4 [])]

secondTree :: Tree Int
secondTree = Node 5 []

listOfTrees = [firstTree, secondTree]
