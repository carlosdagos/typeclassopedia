Typeclassopedia
===============

March 2016.

Source material: [https://wiki.haskell.org/Typeclassopedia](https://wiki.haskell.org/Typeclassopedia)

# FUNCTORS

A computational context to which a function can be applied

### Exercises

- Implement `Functor` instances for `Either e` and `(->) e`

		instance Functor (Either e) where
		    fmap _ (Left x)  = Left x
		    fmap f (Right x) = Right (f x)

		instance Functor ((->) e) where
		    fmap f g = f . g

- Implement `Functor` instances for `((,) e)` and for `Pair`, defined as `data Pair a = Pair a a`

		instance Functor ((,) e) where
		    fmap f (e, x) = (e, f x)

		data Pair a = Pair a a
		instance Functor Pair where
		    fmap f (Pair x y) = Pair (f x) (f y)

- Implement a `Functor` instance for the type `ITree`

		data ITree a = Leaf (Int -> a) | Node [ITree a]
		instance Functor ITree where
		    fmap f (Leaf g)  = Leaf (f . g)
		    fmap f (Node xs) = Node (map f xs)

- Give an example of a type of kind `* -> *` which cannot be made an instance of `Functor` (without using undefined).

Answer:

A type constructor `f a` is a Functor if there's no restriction to any `a`. So it's enough to apply a restriction to a type constructor parameter in order to nullify it's behaviour as a proper Functor.

		:set -XExistentialQuantification
		data OrderedList a = (Ord a) => OrderedList [a]

- Is this statement true or false? "The composition of two `Functor`s is also a `Functor`".

Answer: **True**.

If `F` is a Functor and `G` is a Functor, they have their own `fmap` implementation.

So for a composed functor F (G a), we'll be able to say

		instance Functor (F (G a)) where
		    fmap h x = fmap (fmap h) x

Since the parenthesised fmap lifts h to `G` space, the second fmap will lift `(fmap h)` to `F` space.

## Laws

		fmap id = id                       -- Identity
		fmap (h . g) = (fmap h) . (fmap g) -- Composition

### Exercises

- Although it is not possible for a Functor instance to satisfy the first Functor law but not the second (excluding undefined), the reverse is possible. Give an example of a (bogus) Functor instance which satisfies the second law but not the first.

Answer:

		data Bogus a = Bogus Int a deriving (Show)
		instance Functor Bogus where
		    fmap h (Bogus _ a) = Bogus 1 (h a)

So we have

		fmap h (Bogus 2 "a") = Bogus 1 (h "a")
		fmap (h . g) (Bogus 2 "a") = Bogus 1 ((h . g) "a")             -- By definition
		                           = Bogus 1 (h (g "a"))               -- By composition

		fmap (h . g) (Bogus 2 "a") = (fmap h) . (fmap g) $ Bogus 2 "a"
		                           = fmap h (fmap g (Bogus 2 "a"))     -- By composition
		                           = fmap h (Bogus 1 (g "a"))
		                           = Bogus 1 (h (g "a"))               -- By definition

So we know it obeys the second law. However

		fmap id (Bogus 2 "a") = Bogus 1 (id "a") = Bogus 1 "a"

And it doesn't obey the first law.

- Given a Functor

		-- Evil Functor instance
		instance Functor [] where
		    fmap _ [] = []
		    fmap g (x:xs) = g x : g x : fmap g xs

Which laws are violated?

Answer: Both laws are violated

		fmap id "hey" = "hheeyy"

The second law is also violated

		fmap (h . g) (x:xs) = (h . g) x : (h . g) x : fmap (h . g) xs
		                    = (h (g x)) : (h (g x)) : fmap (h . g) xs
		                    = (h (g x)) : (h (g x)) : fmap h (fmap g xs) -- By composition

		(fmap h) . (fmap g) $ all@(x:xs) = fmap h (fmap g all)
		                                 = fmap h ((g x) : (g x) : (fmap g xs))
		                                 = h (g x) : h (g x) : (g x) : (fmap h (fmap g xs))
		                                 = h (g x) : h (g x) : (g x) : (fmap (h . g) xs) -- Composition

#### Notes

- TIL:  `fmap (h . g) x /= (fmap h) . (fmap g) $ x`, but **SHOULD BE** in Functors
- ALSO: `fmap (h . g) x = fmap h (fmap g x)` -- THIS IS A FUNCTOR LAW

# APPLICATIVES

Effectful computations within a context. They are "idioms" for a speficic domain.

Can also be thought about as a **fixed-length computation** in a certain context.

## Laws

		pure id <*> v = v                             -- Identity
		pure f <*> pure x = pure (f x)                -- Homomorphism
		u <*> pure y = pure ($ y) <*> u               -- Interchange
		             = pure (flip ($) y) <*> u        -- Interchange (equivalent)
		u <*> (v <*> w) = pure (.) <*> u <*> v <*> w  -- Composition
		fmap g x = pure g <*> x = g <$> x             -- <$> is alias for fmap

### Exercises

- (Tricky) One might imagine a variant of the interchange law that says something about applying a pure function to an effectful argument. Using the above laws, prove that

		pure f <*> x = pure (flip ($)) <*> x <*> pure f

Answer:

		pure f <*> x = f <$> x  -- How Applicative relates to Functor
		             = fmap f x -- Definition of <$>

		pure (flip ($)) <*> x <*> pure f = (pure (flip ($)) <*> x) <*> pure f      -- Associativity
		                                 = pure ($ f) <*> (pure (flip ($)) <*> x)  -- Interchange
		                                 = fmap (flip ($) f) (fmap (flip ($)) x)   -- ($ f) = flip ($) f
		                                 = fmap (h f) (fmap h x)                   -- h = flip ($)
		                                 = fmap (h f . h) x                        -- Functor law
		                                 = fmap f x                                -- h f . h = f (*)

`(*)` Proof that if `h = flip ($)`, then `h f . h = f`:

		h f . h = (flip ($) f) . (flip ($))                        -- Definition of h
		        = ((\x -> (\g -> g x)) f) . (\y -> (\g' -> g' y))  -- Definition of flip
		        = (\g -> g f) . (\y -> (\g' -> g' y))              -- Apply first lambda
		        = (\y -> (\g' -> g' y)) f                          -- Reduction
		        = (\y -> f y)                                      -- Definition of lambda
		        = f

- Write `Maybe` as an instance of `Applicative`

		instance Applicative Maybe where
		    pure x        = Just x
		    Just f  <*> x = fmap f x -- Remember Maybe is already a Functor
		    Nothing <*> _ = Nothing

- Determine the correct definition of pure for the `ZipList` instance of `Applicative`. There is only one implementation that satisfies the law relating `pure` and `(<*>)`.

Answer:

		instance Applicative ZipList where
		    pure x = ZipList [x,x..]
		    (ZipList gs) <*> (ZipList xs) = ZipList (zipWith ($) gs xs)

The only `pure` function implementation for ZipList that makes sense is that one, since we must be able to `zipWith` all the elements of the list being applied the applicative functor.

This means that if we create a `pure` ZipList with a single element, our applicative operations will always return 1 element, and since we can't know a priori
the length of the other ZipLists that will be applied, we create an infinite list of the element and let lazy evaluation take care of the rest.

# MONADS

Like `Applicative`, it denotes an "effectful" context. The power of Monad is the fact that we can apply functions to it that are not contextual, but rather are intended to "peek" inside the context, and return (lift -- or fmap) a comptued value in the same context.

### Exercises

- Implement a Monad instance for the list constructor, []. Follow the types!

		class Applicative m => Monad m where
		    return :: a -> m a
		    (>>=)  :: m a -> (a -> m b) -> m b
		    (>>)   :: m a -> m b -> m b
		    m >> n = m >>= \_ -> n
		    fail   :: String -> m a

Answer:

		instance Monad [] where
		    return a = [a]
		    --(>>=) :: [a] -> (a -> [b]) -> [b] (*)
		    xs >>= f = [y | x <- xs, y <- f x]

`(*)`: The signature of bind for [] means that the function will, for each item, need to return a series of computations. Our return type for bind however is [a] so we must flatten the result. This is what `join` means for the monadic [], so `join [[a]] = [a]`, without any data loss.

- Implement a Monad instance for `((->) e)`.

		instance Monad ((->) e) where
		    return = const
		    --(>>=) :: (e -> a) -> (a -> (e -> b)) -> (e -> b)
		    (>>=) g x h = g (h x)

- Implement `Functor` and `Monad` instances for `Free f`, defined as

		data Free f a = Var a | Node (f (Free f a))

_Assume f has a Functor instance_

		instance Functor (Free f) where
		    fmap h (Var a)  = Var (h a)
		    fmap h (Node x) = Node (fmap (fmap h) x)

		instance Monad (Free f) where
		    return = Var
		    (>>=) x y = # TODO

### Intuition

While Applicatives provide us the possibility to operate on a series of computations, these are always restricted. That is, in `u <*> v <*> w`, `u` must be an idiom of two parameters, and then our computations are done. Furthermore, there's no way to use the output of `u <*> v` when applying to `w`, rather `w` is simply applied the previous computation.

In Monads, we can create a sequence of computations of arbitrary length, allowing the next computation to interact with the result of the previous one, and furthermore, like Applicatives, keep the end result within a context (like IO, Maybe, [], so on).

Another explanation is that the computational structure of an Applicative result is fixed, whereas the computational structure of Monadic result can vary depending on the output of the previous computation.

### Exercises

- Implement `(>>=)` in terms of `fmap` (or `liftM`) and `join`.

Answer:

We look at the type of each function

		return :: a -> m a
		(>>=)  :: m a -> (a -> m b) -> m b
		fmap   :: (a -> b) -> f a -> f b
		join   :: m (m a) -> m a

`fmap` looks a lot like the type signagure of `(>>=)`, except it's reversed, but we can do

		flip fmap :: f a -> (a -> b) -> f b

Now if the signature of `(>>=)` says that we must have as a second argument `(a -> m b)`,
then we have that

		flip fmap :: f a -> (a -> f b) -> f b

Then we can say that

		(>>=) :: (Monad m) => m a -> (a -> m b) -> m b
		(>>=) x h = (join . flip) fmap h x

- Now implement `join` and `fmap` (`liftM`) in terms of `(>>=)` and `return`.

		join :: m (m a) -> m a
		join x = x >>= id

		fmap :: (a -> b) -> f a -> f b
		fmap h x = x >>= (\y -> return (h y))

## Laws

		return a >>= k = k a
		m >>= return   = m
		m >>= (\x -> k x >>= h) = (m >>= k) >>= h

Or for the fish operator `(>=>)`

		return >=> g    = g
		g >=> return    = g
		(g >=> h) >=> k = g >=> (h >=> k)

### Exercises

- Given the definition `g >=> h = \x -> g x >>= h`, prove the equivalence of the above laws and the usual monad laws.

Answer:

		return x >>= h  = return a >>= (\y -> h y)         -- expanded
		                = (\x -> return x) >>= (\y -> h y) -- expanded

		return >=> h    = (\x -> return x) >>= h           -- by definition
		                = (\x -> return x) >>= (\y -> h y) -- expanded

		m >>= return    = m a >>= (\x -> return x)         -- expanded
		                = m a >>= return id                -- by definition of id
		                = m a                              -- by definition of return

		m >=> return    = m a >>= (\x -> m x)              -- by definition
		                = m a >>= return id                -- by definition of id
		                = m a                              -- by definition of return

		m >>= (\x -> k x >>= h) = (g >=> h) >=> k          -- by definition

# MONAD TRANSFORMERS

## Definition

		class MonadTrans t where
		    lift :: Monad m => m a -> t m a

## Laws

		lift . return  = return
		lift (m >>= f) = lift m >>= (lift . f)

### Exercises

- What is the kind of `t` in the declaration of `MonadTrans`?

		t :: (* -> *) -> * -> *

The kind of a Monad `m` is `(* -> *) -> *`.

## Composing Monads

The composition of Monads is not always a Monad. Reminder: Applicatives are closed under composition.

### Exercises

- Implement `join' :: M (N (M (N a))) -> M (N a)`, given `distrib :: N (M a) -> M (N a)` and assuming `M` and `N` are instances of `Monad`.

		join    :: (Monad m) => m (m a) -> m a
		join'   :: M (N (M (N a))) -> M (N a)
		distrib :: N (M a) -> M (N a)

Answer

		join' :: M (N (M (N a))) -> M (N a)
		join' x = (join $ x >>= return . distrib) >>= return . join

Proof:

		x                                                 :: M (N (M (N a)))
		x >>= return . distrib                            :: M (M (N (N a)))
		join $ x >>= return . distrib                     :: M (N (N a))
		(join $ x >>= return . distrib) >>= return . join :: M (N a)

# MONADFIX

-- TODO

# FOLDABLE

## Definition

		class Foldable t where
		    fold    :: Monoid m => t m -> m
		    foldMap :: Monoid m => (a -> m) -> t a -> m
		    foldr   :: (a -> b -> b) -> b -> t a -> b
		    foldl   :: (a -> b -> a) -> a -> t b -> a
		    foldr1  :: (a -> a -> a) -> t a -> a
		    foldl1  :: (a -> a -> a) -> t a -> a

### Exercises

- What is the type of `foldMap . foldMap`? Or `foldMap . foldMap . foldMap`, etc.? What do they do?

Answer:

		foldMap :: (Foldable t, Monoid m) =>
		           (a -> m) -> t a -> m
		
		foldMap . foldMap :: (Foldable t, Foldable t1, Monoid m) =>
		           (a -> m) -> t (t1 a) -> m
		
		foldMap . foldMap . foldMap :: (Foldable t, Foldable t1, Foldable t2, Monoid m) =>
		           (a -> m) -> t (t1 (t2 a)) -> m
		
		foldMap . ..(n-1).. . foldMap :: (Foldable t, ..(n-1).., Foldable n, Monoid m) =>
		           (a -> m) -> (t (..(tn a)..)) -> m

By looking at the types `foldMap` allows a function of type `(a -> m)` to operate on a `t a`, that is,
`foldMap` allows the user to operate on a structure's data type without modifying the actual structure.

Ok so here's the actual, proper explanation of what I just tried to say

_Catamorphisms are generalizations of the concept of a fold in functional programming. A catamorphism deconstructs a data structure with an F-algebra for its underlying functor._

_Source: [https://wiki.haskell.org/Catamorphisms](https://wiki.haskell.org/Catamorphisms)_

## Derived folds

		-- Compute the size of any container.
		containerSize :: Foldable f => f a -> Int
		containerSize = getSum . foldMap (const (Sum 1))

For this one the important part is `const (Sum 1)` which is a partially applied function that ignores the parameter and
returns `Sum 1` always. `Sum` is the Monoid with `mappend (Sum x) (Sum y) = Sum (x + y)`.

		-- Compute a list of elements of a container satisfying a predicate.
		filterF :: Foldable f => (a -> Bool) -> f a -> [a]
		filterF p = foldMap (\a -> if p a then [a] else [])

Here the Monoid is `[]`, and `mappend` flattens the list of lists into a list of `a`.

### Exercises

- Implement `toList :: Foldable f => f a -> [a]`.

		toList = foldr (:) mempty

- Implement: `concat`, `concatMap`, `and`, `or`, `any`, `all`, `sum`, `product`, `maximum(By)`, `minimum(By)`, `elem`, `notElem`, and `find`.

		concat :: Foldable t => t [a] -> [a]
		concat = foldMap id

		concatMap :: Foldable t => (a -> [b]) -> t a -> [b]
		concatMap f = foldr (mappend . f) []
		-- OR, to generalise more
		concatMap' :: (Foldable t, Monoid b) => (a -> b) -> t a -> b
		concatMap f = foldr (mappend . f) mempty

		and :: Foldable t => t Bool -> Bool
		and = all (==True) -- We can treat (True, False, &&) as a Monoid with the Identity = True
		                   -- so when evaluating "and []" the result is "True"

In fact, in Haskell this is the Monoid [`All`](https://hackage.haskell.org/package/base-4.8.2.0/docs/src/Data.Monoid.html#All).

		or :: Foldable t => t Bool -> Bool
		or = any (==True)  -- Again, (True, False, ||) as a Monoid with Identity = False
		                   -- So when evaluating "or []" the result is "False"

This is the Monoid [`Any`](https://hackage.haskell.org/package/base-4.8.2.0/docs/src/Data.Monoid.html#Any).

		any :: Foldable t => (a -> Bool) -> t a -> Bool
		any p = getAny . foldMap (Any . p)

		all :: Foldable t => (a -> Bool) -> t a -> Bool
		all p = getAll . foldMap (All . p)

		sum :: (Num a, Foldable t) => t a -> a
		sum = getSum . foldMap Sum

		product :: (Num a, Foldable t) => t a => a
		product = getProduct . foldMap Product

		maximum :: (Ord a, Foldable t) => t a -> a
		maximum = # TODO

		minimum = (Ord a, Foldable t) => t a -> a
		minimum = # TODO

# TRAVERSABLE



		class (Functor t, Foldable t) => Traversable t where
		    traverse  :: Applicative f => (a -> f b) -> t a -> f (t b)
		    sequenceA :: Applicative f => t (f a) -> f (t a)
		    mapM      ::       Monad m => (a -> m b) -> t a -> m (t b)
		    sequence  ::       Monad m => t (m a) -> m (t a)

It is a foldable functor. Only `traverse` or `sequenceA` need to be implemented. By looking at `sequenceA` we can see that it allows us to commute the functors (i.e.: a tree of lists into a list of trees), and by looking at `traverse` we can see that it generalizes Functors: it's an "effectful `fmap`", allowing us to produce a new structure of effectful computations.

### Exercises

- There are at least two natural ways to turn a tree of lists into a list of trees. What are they, and why?

Answer:

It will depend on the type of lists: wether they are of type `[]` or of `ZipList`.

On the former it would make sense to turn the list of trees into all the possible combinations:

		[1,2]------+
		|          |
		[2,3]     [4,5,6]

Turns into

		[ 1----+  ,  1----+  ,  1----+  ,  1----+  ,  1----+  , 1----+ , ... , 2----+
		  |    |     |    |     |    |     |    |     |    |    |    |         |    |
		  2    4     2    5     2    6     3    4     3    5    3    6         3    6  ]

The other way is element-by-element, with `ZipList`:

		ZipList [1, 2]------+
		|                   |
		ZipList [2, 3]    ZipList [4,5,6]

Turns into

		[ 1----+  ,  2----+
		  |    |     |    |
		  2    4     3    5  ]

And the 6 is then disregarded.

- Give a natural way to turn a list of trees into a tree of lists.

		treeOfLists :: [Tree a] -> Tree [a]
		treeOfLists xs = # TODO

This function will work, for example, like so:

		treeOfLists [ 1---+---+ , 2---+    = [1,2]---+------+
		              |   |   |   |   |      |       |      |
		              2   3   4   4   5 ]    [2,4]   [3,5]  [4]

- What is the type of `traverse . traverse`? What does it do?

		traverse
		     :: (Applicative f, Traversable t) => (a -> f b) -> t a -> f (t b)

		traverse . traverse
		     :: (Applicative f, Traversable t, Traversable t1) =>
		        (a -> f b) -> t (t1 a) -> f (t (t1 b))

`traverse` turns a `Traversable` into an `Applicative Functor`, this means that it allows the execution of side-effects within the structure, and returns a structure with the result of those side-effects.

`traverse . traverse` will then allow to run side-effects into nested Traversable structures.

## Laws

		traverse Identity = Identity
		traverse (Compose . fmap g . f) = Compose . fmap (traverse g) . traverse f

### Exercises

- Implement `fmap` and `foldMap` using only the `Traversable` methods.

		fmap :: Traversable f => (a -> b) -> f a -> f b
		fmap h = # TODO

		foldMap :: (Traversable t, Monoid m) => (a -> m) -> t a -> m
		foldMap = # TODO

# CATEGORY

Generalizes the concept of `(->)` as a type constructor.

		class Category cat where
		    id  :: cat a a
		    (.) :: cat b c -> cat a b -> cat a c

Haskell imports an instance of this such as

		instance Category ((->)) where
		    ...

There's another instance

		newtype Kleisli m a b = Kleisli { runKleisli :: a -> m b }

		instance Monad m => Category (Kleisli m) where
		    id = Kleisli return
		    Kleisli g . Kleisli h = Kleisli (h >=> g)

# ARROW

Represents another abstraction of computation. Reflects both input and output, `b \`arr\` c` can be thought of as a computation that takes a `b` as input and produces a `c`. An `Arrow` can represent pure and "effectful" computations.

		class Category arr => Arrow arr where
		    arr    :: (b -> c) -> (b `arr` c)
		    first  :: (b `arr` c) -> ((b, d) `arr` (c, d))
		    second :: (b `arr` c) -> ((d, b) `arr` (d, c))
		    (***)  :: (b `arr` c) -> (b' `arr` c') -> ((b, b') `arr` (c, c'))
		    (&&&)  :: (b `arr` c) -> (b `arr` c')  -> (b `arr` (c, c'))

There's a class constraint of Category, which means that we will have 

		g :: b `arr` c
		h :: c `arr` d
		g >>> h :: b `arr` d

The only methods needed to implement an `Arrow` instance are `arr` and `first`.

See link [https://www.haskell.org/arrows/](https://www.haskell.org/arrows/).

## Laws

		                       arr id  =  id
		                  arr (h . g)  =  arr g >>> arr h
		                first (arr g)  =  arr (g *** id)
		              first (g >>> h)  =  first g >>> first h
		   first g >>> arr (id *** h)  =  arr (id *** h) >>> first g
		          first g >>> arr fst  =  arr fst >>> g
		first (first g) >>> arr assoc  =  arr assoc >>> first g

		assoc ((x,y),z) = (x,(y,z))

## ArrowChoice

Allows `Arrow` computations to be more flexible, as `Arrow` has a fixed-length computation.

		class Arrow arr => ArrowChoice arr where
		    left  :: (b `arr` c) -> (Either b d `arr` Either c d)
		    right :: (b `arr` c) -> (Either d b `arr` Either d c)
		    (+++) :: (b `arr` c) -> (b' `arr` c') -> (Either b b' `arr` Either c c')
		    (|||) :: (b `arr` d) -> (c `arr` d) -> (Either b c `arr` d)

The ArrowChoice class allows computations to choose among a finite number of execution paths, based on intermediate results.

However the downside is that the execution pathds must be know in advance. To solve this, there's `ArrowApply`.

## ArrowApply

		class Arrow arr => ArrowApply arr where
		    app :: (b `arr` c, b) `arr` c

### Exercises

- Implement an alternative “curried” version

		app2 :: b `arr` ((b `arr` c) `arr` c)
		app2 = # TODO

- Finish the implementations

		instance Monad m => ArrowApply (Kleisli m) where
		    app =    -- exercise

		newtype ArrowApply a => ArrowMonad a b = ArrowMonad (a () b)

		instance ArrowApply a => Monad (ArrowMonad a) where
		    return               =    -- exercise
		    (ArrowMonad a) >>= k =    -- exercise

## ArrowLoop

		class Arrow a => ArrowLoop a where
		    loop :: a (b, d) (c, d) -> a b c

		trace :: ((b,d) -> (c,d)) -> b -> c
		trace f b = let (c,d) = f (b,d) in c

Describes arrows that can use recursion to compute results. Desugars the `rec` construct in arrow notation.

Read more: [http://www.staff.city.ac.uk/~ross/papers/fop.html](http://www.staff.city.ac.uk/~ross/papers/fop.html)

# COMONAD

Categorical dual of `Monad`. 

		class Functor w => Comonad w where
		    extract :: w a -> a
		    duplicate :: w a -> w (w a)
		    duplicate = extend id
		    extend :: (w a -> b) -> w a -> w b
		    extend f = fmap f . duplicate

As you can see, `extract` is the dual of `return`, `duplicate` is the dual of `join`, and `extend` is the dual of `(=<<)`.

