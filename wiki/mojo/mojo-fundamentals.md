---
name: mojo-fundamentals
description: Mojo learning note for a Python user, updated for the stable 1.0.0 release (Aug 2026). Covers what Mojo is (Python-compatible systems language by Modular), defining functions with `def` (the `fn` keyword is now deprecated), variables and `var`, ownership and the four argument conventions (`var`/`mut`/default-immutable/`ref`) plus `out`/`deinit` and the `ref` local binding, the `^` transfer sigil, structs and lifecycle traits (`Copyable` is explicit-copy needing `.copy()` vs `ImplicitlyCopyable`, plus `Movable`/move-only/immovable and `@fieldwise_init`), the `__deinit__` destructor (renamed from `__del__`), copy/move constructors as `__init__(*, copy/deinit move)`, traits and generic functions, compile-time parameters in square brackets like `[T, N]`, SIMD/GPU notes, and Python interop (calling into CPython via `Python.import_module`, extracting values with `Int(py=...)`, `PythonObject` aliasing, calling Mojo from Python, GIL, numpy zero-copy). Also covers everyday Python that differs — catching errors with `try`/`except`, `Optional[T]`/`None`, template strings `t"..."` (no f-strings), no `match`, `with`, closures/`lambda` with brace capture — and tooling (install via pixi not pip, `mojo run`/`build`/`repl`/`format`, no `mojo test` (use `TestSuite`), multi-file packages needing `__init__.mojo`, `std.` import prefix). Includes a renaming table for older `fn`/`let`/`borrowed`/`inout`/`owned`/`@value`/`alias`/`__del__`/`__copyinit__`/`StringSlice`/`InlineArray`/`mojo package` syntax. Reach for this when "what does `mut`/`var`/`^` mean", "`[...]` vs `(...)` parameters", "why doesn't `fn` work anymore", "how do I catch an error / return None / interpolate a string in Mojo", "how do I install and run Mojo", or "how do I call NumPy from Mojo".
created: 2026-06-20
updated: 2026-08-14
tags: [mojo, modular, python, systems-programming, ownership, simd, gpu, ai]
---
# Mojo Fundamentals — for Python People Who Want the Whole Picture

## Why This Exists

**Problem.** Mojo markets itself as "Python with superpowers" — and the surface really does look like Python. But the *interesting* parts of Mojo (ownership, compile-time parameters in `[]`, SIMD primitives, GPU kernels, statically-typed `def`) aren't in Python at all. Newcomers see `def main():` and assume the rest is just Python plus speed; they hit `var name: String` or `def foo(mut x: Int)` and the model breaks. This note covers the parts that *aren't* Python.

**Key insight.** Mojo is **a systems language wearing a Python jacket.** It accepts Python-shaped syntax (`def`, indentation, classes, decorators) so the surface feels familiar. But the semantics underneath — static types, ownership keywords, explicit `raises`, compile-time `[]` parameters — are closer to Rust or Zig than to CPython.

**Reach for this when:**
- You see `def main():` and `def foo(mut x: Int):` and want to know what's Python and what's not
- You're confused by `mut`/`var`/`ref`/`^` in argument lists
- You hit `[T: AnyType, N: Int](...)` and want to understand `[...]` vs `(...)`
- Your Python muscle memory won't compile — f-strings, `try`/`except`, `None`, `match`, `**kwargs` (§9)
- You need to actually install, run, or test Mojo and `pip install mojo` / `python foo.py` didn't work (§10)
- You want to call NumPy or PyTorch from Mojo (and vice versa)
- You're choosing between Python (productivity), Mojo (Python-compatible + fast), or Rust (no-Python + safer)

**Don't reach for this when:**
- You want a full reference — go to [mojolang.org/docs](https://mojolang.org/docs/)
- You're starting a non-AI/non-numerical project — Mojo's ecosystem isn't there yet; pick Python, Rust, or Go
- You want production deployment guides — [Modular's MAX platform](https://docs.modular.com/max/) has its own docs

---

## Table of Contents

- [§0. Mental Model — Mojo at a Glance](#0-mental-model--mojo-at-a-glance)
- [§1. Hello World and the Shape of `def`](#1-hello-world-and-the-shape-of-def)
- [§2. Variables, Types, and Mutability](#2-variables-types-and-mutability)
- [§3. Ownership and Argument Conventions](#3-ownership-and-argument-conventions)
- [§4. Structs and the Value Lifecycle](#4-structs-and-the-value-lifecycle)
- [§5. Traits — Mojo's Type Contracts](#5-traits--mojos-type-contracts)
- [§6. Compile-Time Parameters — the `[...]` Square Brackets](#6-compile-time-parameters--the--square-brackets)
- [§7. SIMD, Tensors, and Hardware Awareness](#7-simd-tensors-and-hardware-awareness)
- [§8. Python Interop](#8-python-interop)
- [§9. Everyday Python That Behaves Differently](#9-everyday-python-that-behaves-differently)
- [§10. Tooling — Install, Run, Test, Ship](#10-tooling--install-run-test-ship)
- [Trade-offs](#trade-offs)
- [Common Pitfalls](#common-pitfalls)
- [Decision Table](#decision-table)
- [References](#references)
- [See Also](#see-also)

---

## 0. Mental Model — Mojo at a Glance

> *TL;DR — Python-syntax on the outside, Rust-style semantics on the inside. One keyword for functions (`def`), with types and explicit error-propagation. `(...)` for runtime args, `[...]` for compile-time parameters. Ownership without lifetimes.*

> **Version note.** This note tracks **Mojo 1.0.0**, the first stable release ([shipped 2026-08-11](https://mojolang.org/docs/changelog)). The 1.0.0 milestone stabilises the language surface described here and consolidates several older spellings — `fn` is now deprecated in favour of `def`, the destructor dunder is `__deinit__` (not `__del__`), and a few stdlib types were renamed. See the [renaming table](#older-syntax-youll-see-in-tutorials-renamed-or-removed) in §1 for the migration you'll need when reading pre-1.0 blog posts and tutorials.

Mojo is built by [Modular](https://www.modular.com) (Chris Lattner — same person behind [LLVM](https://llvm.org) and [Swift](https://www.swift.org)) to give AI/numerical Python the speed of C while keeping the Python surface. Its design has three big claims:

1. **Python syntax is a starting point.** A `.mojo` file with `def main(): print("hi")` runs. You can mix in NumPy via `Python.import_module`.
2. **Static typing and ownership underneath.** Functions need types, errors must declare `raises`, and values have a single owner. The compiler produces code as fast as Rust or C++.
3. **First-class compile-time programming.** SIMD width, tensor shape, dtype — all expressed as compile-time parameters in `[]`, specialised per call.

```mermaid
flowchart TD
    Py["Pure Python\n(dynamic, runtime-checked)"] --> Mojo["Mojo (def with types)\nstatic types, raises, var/mut/^ ownership"]
    Mojo --> Param["Parametric Mojo\n[compile-time T, N, dtype]"]
    Param --> HW["GPU / SIMD kernels\ntiling, vector width as parameters"]
```

You start with familiar Python *syntax* but the language underneath is a systems language — statically typed, with explicit ownership and explicit error propagation.

---

## 1. Hello World and the Shape of `def`

> *TL;DR — every function uses `def`. Type annotations are required on parameters; return types must be declared if the function returns a value; raising errors requires `raises` on the signature.*

### Hello world

Every Mojo program needs a `main()`:

```mojo
def main():
    print("Hello, world!")
```

Save as `hello.mojo`, run with `mojo hello.mojo`. Looks exactly like Python.

### `def` with types

Beyond `main`, `def` requires you to annotate parameters and (when returning a value) the return type:

```mojo
def greet(name: String) -> String:
    return "Hello, " + name + "!"

def add(a: Int, b: Int) -> Int:
    return a + b
```

This is **not** the loose, Python-style `def` you may know. Mojo's `def` is statically typed under the hood. The official rules:

- **Parameter types required.** `def foo(x):` won't compile; you need `def foo(x: Int):`.
- **Return types declared.** Functions that return a value need `-> Type`. `-> None` (or omitting it) means "returns nothing."
- **`raises` is explicit.** Functions can't throw unless declared: `def parse(s: String) raises -> Int: ...`. Non-raising callers must handle errors from raising callees.
- **Typed errors supported.** `def validate(x: Int) raises ValidationError -> Int: ...` lets you constrain the error type.
- **Overloading.** Since types are mandatory, multiple-type support means defining overloads (same name, different parameter types).

### `def` is the one function keyword — `fn` is deprecated

If you've read older Mojo material you'll have seen `fn` everywhere, often paired with `def` as "the strict one vs the loose one." **That split is gone.** As of 1.0.0, `def` *is* the strict, statically-typed function keyword documented above, and `fn` is deprecated — the docs say it "should not be used for new function declarations." Write `def`; if you inherit `fn` code it still compiles for now, but it's on the way out.

### Older syntax you'll see in tutorials (renamed or removed)

Most Mojo content online predates 1.0.0. When a snippet won't compile, it's usually one of these renames — the *concepts* in this note are unchanged, only the spelling moved:

| Old (pre-1.0) | Now (1.0.0) | What changed |
|---|---|---|
| `fn foo():` | `def foo():` | `fn` deprecated; `def` is the single function keyword |
| `let x = 5` | `var x = 5` (or implicit) | `let` was removed; `var` (or bare assignment) declares |
| `borrowed x` | `x` (no keyword) | immutable borrow is now the unnamed default (§3) |
| `inout x` | `mut x` | mutable-borrow convention (§3) |
| `owned x` | `var x` | take-ownership convention (§3) |
| `@value` | `@fieldwise_init` + `Copyable`/`Movable` | the old catch-all decorator split into an init generator plus explicit trait conformance (§4) |
| `alias X = ...` | `comptime X = ...` | compile-time binding keyword (§6) |
| `def __del__(...)` | `def __deinit__(...)` | destructor dunder renamed for parity with `__init__`; `__del__` still works but is deprecated (§4) |
| `fn __copyinit__` / `fn __moveinit__` | `__init__(out self, *, copy: Self)` / `__init__(out self, *, deinit move: Self)` | copy/move constructors are now named `__init__` overloads (§4) |
| `SIMD[dt, size=N]` | `SIMD[dt, N]` or `SIMD[dt, length=N]` | SIMD's second parameter renamed `size` → `length`; positional use is unchanged (§7) |
| `ImplicitlyDeletable` / `ImplicitlyDestructible` | `Deinitable` | the "has a destructor" trait was renamed (§6) |
| `StringSlice` | `StringSpan` | non-owning string view renamed; `StringSlice` kept as a compatibility alias (§2) |
| `InlineArray` | `Array` | fixed-size array renamed; `InlineArray` kept as a compatibility alias (§2) |
| `from math import sqrt` | `from std.math import sqrt` | the stdlib now lives under the `std.` package (§10) |
| `mojo package` → `.mojopkg` | `mojo precompile` → `.mojoc` | the package-build command and artifact were renamed (§10) |

---

## 2. Variables, Types, and Mutability

> *TL;DR — `var x = 5` declares; you can omit `var` and let inference handle it. Types are static — set on first assign and fixed for the binding's lifetime.*

### Declaring variables

```mojo
var x: Int = 10          # explicit type and value
var sum: Int             # declared, initialised later
sum = x + x

y = 5                    # implicit var, type inferred as Int
z = "hello"              # implicit var, inferred as String
```

The compiler infers a type from the first assignment and **freezes it for the binding's lifetime**:

```mojo
x = 10
x = "foo"                # ERROR: Cannot convert "StringLiteral" value to "Int"
```

This is the single biggest difference from Python — bindings are statically typed even when types aren't written.

### Primitive types you'll meet

| Type | What it is |
|---|---|
| `Int` | machine-width integer (typically 64-bit) |
| `UInt` | unsigned variant |
| `Float64`, `Float32`, `Float16` | floats of given width |
| `Bool` | `True` / `False` |
| `String` | heap-allocated, mutable UTF-8 string |
| `StringLiteral` | compile-time string literal (`"hi"` in source) |
| `StringSpan` | non-owning *view* into existing UTF-8 — a zero-copy substring (`StringSlice` is a compat alias) |
| `SIMD[DType.float32, 4]` | a 4-wide vector of `f32` — the building block for vectorised math |
| `List[T]` | growable array (Python-style, heap) |
| `Array[T, N]` | **fixed-size**, compile-time-length array (`InlineArray` is a compat alias) |
| `Dict[K, V]` | hash map (Python-style) |

`SIMD` is the standout — it's not a library type but a built-in for direct hardware vector control. More in §7.

Two of these have no Python analogue and are worth flagging:

- **`StringSpan` is a view, not a copy.** Python's `s[1:5]` always allocates a new string; a `StringSpan` borrows into the original's bytes (zero-copy), so it's cheap but tied to the source's lifetime.
- **`Array` vs `List` is a real choice.** `Array[Int, 3]` has its length baked in at compile time (stack-friendly, no allocation); `List[Int]` grows at runtime (heap). Python only gives you the growable `list`; Mojo makes you pick. A bare literal like `[1, 2, 3]` infers `List` when unannotated, but coerces to `Array` if the binding is typed `Array[Int, 3]`.

### Control flow

Identical to Python at the surface:

```mojo
def classify(n: Int) -> String:
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    else:
        return "positive"

def sum_of_squares(n: Int) -> Int:
    var total = 0
    for i in range(n):
        total += i * i
    return total

def countdown(n: Int):
    while n > 0:
        print(n)
        n -= 1
```

Indentation, `:`, `range()` — all Python. The everyday subset works.

---

## 3. Ownership and Argument Conventions

> *TL;DR — every value has exactly one owner. Function arguments use one of four conventions: default (immutable borrow), `mut` (mutable borrow), `var` (take ownership), or `ref` (parametric mutability). The `^` sigil transfers ownership.*

This is Mojo's most Rust-like aspect — and the part that confuses Python migrants the most.

### The four argument conventions

| Convention | What the function gets | Caller's value after |
|---|---|---|
| *(default — no keyword)* | immutable reference | unchanged |
| `mut` | mutable reference | may be modified |
| `var` | takes ownership | gone (or copied, see below) |
| `ref` | reference with parametric mutability | depends on the parametric annotation |

These four are what you reach for on ordinary arguments. Two more conventions exist for a narrower job — you'll only write them on `self`:

- **`out`** — the value starts *uninitialised* and must be filled in before the function returns. This is what `__init__(out self, ...)` uses (§4), and it also names function results.
- **`deinit`** — the value starts *initialised* and is left *uninitialised* on return. This marks a destructor or a consuming move, i.e. `__deinit__(deinit self)` (§4).

(One naming note for Rust migrants: Mojo spells "take ownership" as `var`, not `owned` — `owned` was the pre-1.0 keyword.)

### Default — borrowed, immutable

```mojo
def greet(name: String):           # name is an immutable borrow
    print("Hello, " + name)

def main():
    var n = String("Alice")
    greet(n)                       # n still usable after
    print(n)
```

No keyword needed. The function receives a read-only reference. Cheap for large values — no copy, no move.

### `mut` — mutable borrow

```mojo
def add(mut x: Int, y: Int):
    x += y

def main():
    var a = 1
    var b = 2
    add(a, b)
    print(a)                       # 3
```

The function can mutate `x`; the caller sees the change.

### `var` — take ownership

```mojo
def consume(var s: String):
    print(s)
    # s dropped at end of function

def main():
    var name = String("Alice")
    consume(name^)                 # transfer ownership
    # name is now uninitialised — using it is a compile error
```

`var` *can* receive ownership, but **only if you supply the `^` sigil**. Without `^`, Mojo tries to *copy* the value (and fails to compile if the type isn't copyable).

### `^` — the transfer sigil

`^` is the postfix operator that explicitly transfers ownership:

```mojo
list.append(name^)                 # name's lifetime ends here
# name can't be used afterwards
```

After `^`, the source variable is *uninitialised*. The compiler tracks this and rejects further uses. This is Rust's `move` semantics with explicit punctuation rather than implicit moves on assignment.

### `ref` — parametric mutability

The first three conventions all *commit* to a mutability: default is "always immutable," `mut` is "always mutable," `var` is "I take ownership." But sometimes you need a function that **works for both** — without writing it twice.

The classic example is an accessor like "give me a reference to the first element of a list":

- If the caller has an *immutable* list, the function should return an *immutable* reference (read-only).
- If the caller has a *mutable* list, the function should return a *mutable* reference (so the caller can assign through it).

If you commit to one or the other, you either lose write capability for mutable callers or lock out immutable callers. `ref` solves this by **making the mutability itself a compile-time parameter**.

#### Walking through the example

```mojo
def get_first[is_mut: Bool](ref [is_mut] container: List[Int]) -> ref [is_mut] Int:
    return container[0]
```

There are two layers of brackets, doing different jobs:

| Bracket | What it is |
|---|---|
| `[is_mut: Bool]` *(after the function name)* | a **compile-time parameter** — see §6. The caller fills in `True` or `False`. |
| `ref [is_mut]` *(before the argument/return)* | a **reference whose mutability equals `is_mut`** |

Read it left-to-right:
- `def get_first[is_mut: Bool]` — a function parameterised by a compile-time `Bool` called `is_mut`.
- `(ref [is_mut] container: List[Int])` — takes a reference to a `List[Int]` whose mutability follows `is_mut`. If the caller passes `True`, it's a `mut` reference; if `False`, an immutable one.
- `-> ref [is_mut] Int` — returns a reference to an `Int` with the *same* mutability. The shared `is_mut` ties input and output together.

The compiler generates *two* specialised versions of this function: one for `is_mut = True` (mutable in, mutable out) and one for `is_mut = False` (immutable in, immutable out). Same source, no duplication.

#### Why the inference does the work for you

You almost never write `[is_mut]` at the call site. Mojo infers it:

```mojo
var nums: List[Int] = [1, 2, 3]

var first_ref = get_first(nums)    # nums is mutable → is_mut = True inferred
first_ref = 99                      # OK — mutable reference, can assign

# vs reading through an immutable handle
def show(numbers: List[Int]):       # numbers is immutable (default)
    var x = get_first(numbers)      # is_mut = False inferred
    # x = 99 would fail — immutable reference
    print(x)
```

The caller's context decides which version gets used. From outside, `get_first` "just works" like any normal function — the polymorphism is invisible.

#### Where you'll meet this

- **Indexing methods** on collections (`__getitem__`-style) — `list[i]` needs to work for both reading and writing.
- **Iterators** that yield references into a collection.
- **View / slice types** that should be valid for both read-only views and writable views of the same data.

You rarely write `ref` yourself for application code — the four conventions (default / `mut` / `var` / explicit copy) cover almost everything. But every collection you reach for already uses `ref` internally, and reading stdlib code will go smoother once you can parse the syntax.

### Holding a returned reference — the `ref` binding

`ref` isn't only an argument convention; it's also a **local binding form**. This matters the moment you index a collection. In Python, `x = mylist[0]` aliases the element for free. In Mojo, `var x = mylist[0]` *copies* it (or errors if the element type isn't copyable) — and mutating `x` won't touch the list. To alias instead of copy, bind with `ref`:

```mojo
var nums = [10, 20, 30]

var a = nums[0]        # a is a COPY — a = 99 leaves nums unchanged
ref b = nums[0]        # b is an alias into the list
b = 99                 # writes through — nums is now [99, 20, 30]
```

This is the piece that closes the `get_first` example above: the returned `ref` only stays a reference if the caller catches it with a `ref` binding.

### Argument exclusivity

Same rule as Rust: if a function receives a mutable reference to a value, it can't receive any other reference (mutable or immutable) to the same value.

```mojo
def double(mut x: Int, y: Int): ...

var a = 5
double(a, a)                       # ERROR: a is both mut and read
```

This is enforced at compile time. The exception: trivially copyable types like `Int` and `Bool` are passed by copy in registers, so the rule doesn't bite them as often.

### Why this matters — and how it differs from Rust

Mojo's ownership system is **deliberately simpler than Rust's**. The two big simplifications:

- **No explicit lifetimes.** Mojo uses *origins* (an internal tracking mechanism) but doesn't make you write `<'a>` annotations. The compiler infers everything.
- **Default is immutable borrow, not move.** Python-style call semantics work because passing a `String` doesn't move it — you'd have to explicitly write `s^` or `var s:` to consume.

The result: most Python-shaped code "just works" without thinking about ownership. The ceremony only kicks in when you write `var` parameters, `mut`, or `^` — and at those points you're already opting into systems-level thinking.

---

## 4. Structs and the Value Lifecycle

> *TL;DR — `struct` with `var` fields, `__init__` constructor, optional `@fieldwise_init` shortcut. Conform to `Copyable`/`Movable` traits to auto-generate copy/move logic.*

### A basic struct

```mojo
struct MyPair:
    var first: Int
    var second: Int

    def __init__(out self, first: Int, second: Int):
        self.first = first
        self.second = second
```

Reading the new bits:
- **`var` on fields is required** — no auto-typed fields like in Python.
- **`out self`** — declares `self` as a mutable reference that starts uninitialised and must be filled in before the constructor returns.

### `@fieldwise_init` — the boilerplate shortcut

```mojo
@fieldwise_init
struct MyPair:
    var first: Int
    var second: Int
```

The decorator auto-generates `__init__(out self, first: Int, second: Int)`. Use it when your constructor is just "store the args."

### Methods

```mojo
@fieldwise_init
struct Counter:
    var n: Int

    def get(self) -> Int:           # default — borrowed, read-only
        return self.n

    def increment(mut self):        # mutable borrow
        self.n += 1

    @staticmethod
    def from_zero() -> Counter:
        return Counter(0)
```

The `self` parameter follows the same conventions as any other argument (§3) — `self` borrowed by default, `mut self` for mutating methods.

### Copy and move via traits

**By default a Mojo struct can be neither copied nor moved.** This is the biggest surprise for a Python dev: `var b = a` on your own struct is a *compile error* until you opt in. You opt in by conforming to traits, and the compiler generates the machinery:

```mojo
@fieldwise_init
struct Counter(Copyable):
    var n: Int
```

But "copyable" in Mojo does **not** mean "copies silently like a Python assignment." There are two distinct copy traits, and the difference is exactly the trap:

| Trait | What it enables | `var b = a` does… |
|---|---|---|
| `Movable` | move constructor — the `^` transfer works | error (no copy) — must write `a^` to move |
| `Copyable` | **explicit** copy — you call `.copy()` | **error** — you must write `var b = a.copy()` |
| `ImplicitlyCopyable` | **implicit** copy — assignment/passing copies for you | copies, exactly like Python |

So a struct marked only `Copyable` still rejects `var b = a`; the compiler tells you to call `a.copy()`. That's deliberate — copies of heavyweight types are made *visible* in the source. Only `ImplicitlyCopyable` (which the built-in scalars like `Int` use) restores the Python reflex of "assignment just copies." `Copyable` implies `Movable`; `ImplicitlyCopyable` implies `Copyable`.

```mojo
var i = 5
var i2 = i            # OK — Int is ImplicitlyCopyable

@fieldwise_init
struct Big(Copyable):  # explicit-copy only
    var data: List[Int]

var a = Big([1, 2, 3])
var b = a              # ERROR: use a.copy()
var c = a.copy()       # OK — the copy is spelled out
```

### Move-only and immovable types

Because the traits are opt-in, you can also *deliberately* leave them off:

- **Move-only** (`Movable` but not `Copyable`): the value can be transferred with `^` but never duplicated. This is how unique-ownership types like `OwnedPointer` work — a Python dev has no analogue for "a value you literally cannot copy."
- **Immovable** (neither trait): the value is pinned in place and can be neither copied nor moved (e.g. `Atomic`). It can only be borrowed.

### The generated constructors — and when you must write them by hand

For plain data structs the compiler writes the copy/move constructors for you, so you rarely see their spelling. When you *do* hand-write one (1.0.0 renamed these — pre-1.0 code uses `__copyinit__`/`__moveinit__`):

```mojo
def __init__(out self, *, copy: Self): ...        # copy constructor
def __init__(out self, *, deinit move: Self): ...  # move constructor
```

**You are forced to write these when your struct holds a raw `UnsafePointer`** (or points at itself). The compiler's default copy would duplicate the *pointer*, giving you two structs aliasing the same buffer (double-free / dangling on destruction); the default move breaks self-references. So a buffer/arena/linked-node type needs a custom copy constructor that allocates fresh memory, and often a custom move. This is the one place the friendly "the compiler handles it" story stops.

### Destructor: `__deinit__`

Mojo calls `__deinit__` automatically when a value's owner goes out of scope. You only define one if you need custom cleanup:

```mojo
struct FileHandle:
    var fd: Int

    def __init__(out self, path: String):
        self.fd = open_file(path)

    def __deinit__(deinit self):
        close_file(self.fd)
```

The `deinit self` convention marks the destructor — `self` is alive on entry, uninitialised on exit (§3).

> **Renamed in 1.0.0.** The destructor dunder was `__del__` before 1.0.0; it's now `__deinit__`, matching `__init__`. `__del__` still compiles but emits a deprecation warning with a fix-it to rename it.

---

## 5. Traits — Mojo's Type Contracts

> *TL;DR — traits look like Python ABCs but resolve at compile time. Conform with `struct Foo(TraitA, TraitB):`. Bound generic params with `[T: TraitA]`.*

### Defining a trait

```mojo
trait Quackable:
    def quack(self):
        ...
```

`...` is the "body must be implemented by conforming types" placeholder. Compare to Rust traits or Python's `abc.ABC`.

### Conforming

```mojo
@fieldwise_init
struct Duck(Copyable, Quackable):
    def quack(self):
        print("Quack")
```

Just list traits in parentheses after the struct name. The compiler checks that every required method is implemented.

### Generic functions with trait bounds

Two equivalent forms:

```mojo
# Named-parameter form
def make_it_quack[DuckType: Quackable](maybe_a_duck: DuckType):
    maybe_a_duck.quack()

# Shorthand
def make_it_quack2(maybe_a_duck: Some[Quackable]):
    maybe_a_duck.quack()
```

Use the named form when you need to constrain *two* arguments to be the same type (`def pair[T: Quackable](a: T, b: T)`).

### Trait inheritance (refinement)

```mojo
trait Bird(Animal):
    def fly(self):
        ...
```

A `Bird` is required to implement everything `Animal` does, plus `fly`. Anywhere `Animal` is required, a `Bird` is acceptable.

### Trait composition with `&`

Combine traits ad-hoc without naming the combination:

```mojo
def quack_and_go[T: Quackable & Flyable](thing: T):
    thing.quack()
    thing.fly()
```

Or name it for reuse: `comptime DuckLike = Quackable & Flyable`. Unlike refinement, *any* type conforming to both component traits automatically satisfies the composition — no need to declare the composed trait explicitly.

### Built-in traits worth knowing

| Trait | Provides | Typical use |
|---|---|---|
| `Copyable` | copy constructor | value-types you want to clone |
| `Movable` | move constructor | almost everything |
| `Defaultable` | `__init__(out self)` | "give me a default instance" |
| `Sized` | `__len__()` → `len(x)` works | collections |
| `Stringable` (`Writable`) | `print(x)` works | anything you'd display |
| `Comparable`, `Equatable`, `Hashable` | `<`, `==`, hashing | map keys, sorting |
| `Intable` | `__int__()` → `Int(x)` | numeric coercion |
| `AnyType` | the universal base | unconstrained generic parameters |

---

## 6. Compile-Time Parameters — the `[...]` Square Brackets

> *TL;DR — Mojo calls compile-time values "parameters" and runtime values "arguments." `[...]` for parameters, `(...)` for arguments. The compiler generates a specialised version per parameter combination — same idea as Rust monomorphization.*

### Parameters vs arguments

```mojo
def repeat[count: Int](msg: String):
    comptime for i in range(count):
        print(msg)

repeat[3]("Hello")
```

`[count: Int]` — compile-time, must be known when the call is compiled. `(msg: String)` — runtime, evaluated each call.

This terminology trips people up: in most languages "parameter" and "argument" are nearly synonyms. In Mojo, they're a hard distinction the syntax makes visible.

### Parametric structs

```mojo
struct FixedBuffer[size: Int]:
    var data: List[Int]
    # ... using `size` as a compile-time constant
```

`FixedBuffer[4]` and `FixedBuffer[8]` are *different types* (just like `Vec<i32>` vs `Vec<f64>` in Rust).

### Type parameters

```mojo
struct GenericArray[T: Copyable & Deinitable]:
    var data: UnsafePointer[T]
    var size: Int
```

`[T: ...]` is "any type satisfying these traits" — Mojo's equivalent of `<T>` plus trait bounds. (`Deinitable` is the 1.0.0 trait meaning "has a destructor"; older tutorials spell it `ImplicitlyDeletable` / `ImplicitlyDestructible`.)

### Parameter inference

You usually don't write `[...]` at call sites; the compiler infers:

```mojo
def rsqrt[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return 1 / sqrt(x)

var v: Scalar[DType.float16] = 4
rsqrt(v)                        # dt inferred as DType.float16
```

### When to use parameters vs arguments

**Parameters** (`[]`): values that should drive code specialisation — types, dimensions, SIMD widths, dtypes, dispatch flags.

**Arguments** (`()`): values varying at runtime — the data you're operating on.

Hardware-aware code lives in this distinction. `SIMD[DType.float32, 4]` is parameterised so the compiler emits the exact AVX/NEON instructions for that width/dtype — no runtime dispatch.

### `comptime` blocks

The `comptime` keyword forces evaluation at compile time:

```mojo
comptime for i in range(4):       # unrolled completely
    sum += data[i]

comptime if some_compile_time_flag:
    fast_path()
else:
    slow_path()
```

---

## 7. SIMD, Tensors, and Hardware Awareness

> *TL;DR — `SIMD[dtype, length]` is a built-in vector type. A single-lane SIMD is a `Scalar` — and `Int` is just `Scalar[DType.int]` (i.e. `SIMD[DType.int, 1]`). Tensor work uses the `LayoutTensor` family. GPU programming has dedicated language support.*

This is the "superpowers" half of the Python-with-superpowers tagline.

### SIMD is built-in, not a library

```mojo
var v: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
var w: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](10.0, 20.0, 30.0, 40.0)
var sum = v + w                   # 4 lanes added in one instruction
```

The compiler emits one vector-add instruction (`vaddps` on x86 AVX, `fadd` on NEON). No library, no intrinsic call — just an arithmetic operator.

The scalar types are the degenerate one-lane case: `Scalar` is `SIMD[length=1]`, `Float32` is `Scalar[DType.float32]`, and `Int` is `Scalar[DType.int]`. Mojo's numeric stack is SIMD all the way down — a plain `Int` is a one-wide integer vector.

> **Renamed in 1.0.0.** SIMD's second parameter was called `size`; it's now `length`. Positional use (`SIMD[DType.float32, 4]`) is unchanged, but the keyword form is now `length=4` — passing `size=4` is an error, and reading `v.size` warns.

### Why this is a Python improvement

In NumPy, `a + b` on vectors works — but each operation walks Python's bytecode interpreter once. Mojo's `SIMD` operations compile to a single CPU instruction. Numerical code that bottlenecks on Python's per-call overhead becomes cache-line-bound or vector-unit-bound instead.

### GPU programming

Mojo has first-class GPU primitives (kernels, layouts, async memory copies, warp-level intrinsics). The full topic deserves its own note — see the [GPU section of the manual](https://mojolang.org/docs/manual/gpu/architecture/). For Python users, the takeaway: writing a custom CUDA kernel in Mojo is roughly as ergonomic as writing a `numba.cuda` kernel, but the rest of your program is Mojo too, so you don't pay the Python-host overhead between kernels.

### Tensors and layouts

`LayoutTensor` and `TileTensor` are the multidimensional-array types optimised for GPU kernels. They expose tile shape, memory layout, and access patterns as compile-time parameters, letting the compiler generate optimal access code. PyTorch/JAX users can think of this as TPU/GPU-style explicit-layout tensors with no Python overhead.

---

## 8. Python Interop

> *TL;DR — `from std.python import Python; np = Python.import_module("numpy")`. NumPy arrays, dataclasses, anything pip-installed — all reachable from Mojo. Getting native values back out uses `Int(py=obj)`-style constructors, and the interop call must live inside a function (no top-level code).*

This is the killer feature that makes Mojo viable as a *gradual* migration from Python.

### Calling Python from Mojo

```mojo
from std.python import Python

def main() raises:
    var np = Python.import_module("numpy")   # the CPython numpy you pip-installed
    var ar = np.arange(15).reshape(3, 5)
    print(ar)
    print(ar.shape)
```

That `numpy` is the Python NumPy you already have installed. Mojo loads CPython under the hood (as `libpython`) and calls into it.

Two things a Python dev trips over immediately:

- **The import path is `std.python`,** not `python`. (Pre-1.0 tutorials wrote `from python import Python`.)
- **`import_module` must live *inside* a function.** Mojo has no top-level executable code, so a module-level `import numpy` / `Python.import_module(...)` won't work — put it in `main()` or a helper. This is also why `def main()` needs **`raises`**: interop can throw.

### Getting values back out — `PythonObject` isn't a Mojo value

`import_module` and every Python call return a `PythonObject`, a boxed handle to a CPython object. You can keep chaining Python calls on it (`ar.reshape(...)`, `ar.shape`), but **most typed Mojo APIs won't accept a `PythonObject`** — you have to convert it back:

```mojo
from std.python import Python

def main() raises:
    var os = Python.import_module("os")
    var n_py = os.cpu_count()          # PythonObject
    var n = Int(py=n_py)               # -> native Int via the `py=` constructor
    print(n * 2)                        # now it's a real Mojo Int
```

`Int`, `Float64`, `Bool`, and `String` all have a keyword-only `py=` constructor (the `ConvertibleFromPython` trait). Going the *other* way, only those four primitives auto-convert Mojo→Python; Mojo collections do **not** marshal automatically — build Python containers explicitly with `Python.list(...)`, `Python.dict()`, `Python.tuple(...)`, or `Python.evaluate("...")`.

### `PythonObject` keeps Python's reference semantics

Everything else in this note is value-semantic (§2–§4). `PythonObject` is the exception — it aliases like a normal Python variable:

```mojo
var a = Python.list(1, 2, 3)
var b = a          # NOT a copy — b and a are the same CPython list
b.append(4)
print(a)           # [1, 2, 3, 4]  — a changed too
print(a is b)      # True — `is` works, same identity
```

So inside interop you're back in Python's mutable-shared-object world; the ownership rules only re-apply once you convert to native Mojo values.

The `raises` keyword on `def main()` is important — Python interop can throw, so the function must declare it can raise.

### Calling Mojo from Python

The reverse direction — build Mojo as a Python extension module and `import` it from regular Python — is what makes "speed up one hot function" possible. **In 1.0.0 this is explicitly early-development/beta**: the manual warns to "expect a lot of changes to the API and ergonomics." Practical limits to know before you lean on it:

- You export functions via a `PythonModuleBuilder` and build with `mojo build --emit shared-lib`.
- An exported function takes **at most 6 `PythonObject` arguments**, and native Mojo `*args`/`**kwargs` aren't supported at the boundary.
- The `abi("C")` entry point **can't be `raises`**.

For a Python dev this undercuts the tidy "just move the bottleneck to Mojo" pitch — it works, but the ergonomics are still rough. The *calling Python from Mojo* direction (above) is the mature one.

### The performance boundary — GIL and zero-copy

Two realities govern whether interop actually pays off:

- **The GIL still applies.** Mojo loads unmodified CPython as `libpython`, so every interop call holds Python's Global Interpreter Lock. Mojo threads calling into Python serialise just as Python threads do — interop buys you native *single-thread* speed, not free parallelism.
- **Bulk data can be shared without copying.** Marshalling a big array element-by-element through `PythonObject` is slow. For NumPy, grab the raw buffer once and index it natively:

  ```mojo
  var np = Python.import_module("numpy")
  var arr = np.arange(1_000_000, dtype=np.int64)
  var ptr = arr.ctypes.data.unsafe_get_as_pointer[DType.int64]()  # zero-copy view
  var total: Int64 = 0
  for i in range(1_000_000):
      total += ptr[i]                                              # native speed, no CPython per element
  ```

  (The newer `python.numpy` `from_numpy_array`/`copy_to_numpy_array` helpers exist only in **nightly**, not 1.0.0 — use the `ctypes.data` route on stable.)

### When this is worth it

- You have NumPy/PyTorch code that's bottlenecked on a small kernel — write the kernel in Mojo, leave the rest in Python.
- You want CPython library access (e.g., `requests`, `pandas`) but want some functions to run native-speed.
- You're testing Mojo incrementally without committing the whole codebase.

### When this isn't worth it

- You're not on the AI/numerical path — Python interop adds the CPython runtime to your binary, which kills the "tiny systems-language binary" pitch.
- You're shipping to environments where CPython isn't available.

---

## 9. Everyday Python That Behaves Differently

> *TL;DR — the constructs you use on line one of every Python file — catching errors, `None`, f-strings, `match`, `with`, `lambda`, `**kwargs` — all exist but with sharp edges. This section is the "muscle-memory won't compile" list.*

The earlier sections cover the big new *concepts*. This one covers small, high-frequency Python habits that look identical but differ — the stuff that stops you writing basic code on day one.

### Catching errors — `try` / `except`

§1 showed how to *raise* (`raises` on the signature). Catching looks like Python but with real restrictions:

```mojo
def parse(s: String) raises -> Int:
    if s == "":
        raise "empty input"       # a bare string is auto-wrapped in an Error
    return atol(s)

def main():
    try:
        var n = parse("")
        print(n)
    except e:                     # bind with `except e:` — NOT `except SomeError as e:`
        print("failed:", e)
    else:
        print("ok")               # else/finally work like Python
    finally:
        print("done")
```

The traps, all different from Python:

- **One `except` per `try`.** You can't stack `except ValueError:` / `except KeyError:` — there's a single clause.
- **No type matching.** `except SomeError as e:` isn't supported; the error type is inferred, and you bind it with `except e:` (or omit the name). Mojo errors are plain values, not a class hierarchy you match on.
- **`raise "text"`** auto-wraps the string in an `Error`. There's no `raise ValueError(...)` taxonomy.
- Errors are **values, not stack-unwinding exceptions** — cheap, but there's no traceback by default (set `MODULAR_DEBUG=stack-trace-on-error` to get one).

### `None` isn't free — `Optional[T]`

In Python any variable can be `None`. In statically-typed Mojo, `var x: Int = None` is a type error. "Might be absent" is its own type, `Optional[T]`:

```mojo
from std.collections import Optional

def first_even(nums: List[Int]) -> Optional[Int]:
    for n in nums:
        if n % 2 == 0:
            return n
    return None                    # allowed: Optional[Int] accepts None

def main():
    var r = first_even([1, 3, 4])
    if r:                          # __bool__ tests presence
        print(r.value())           # unwrap — but see the trap below
    print(first_even([1, 3]).or_else(-1))   # -1 default if empty
```

The trap for a Python dev: **`.value()` on an empty `Optional` aborts the whole process** (not a catchable exception). Guard with `if opt:` / `opt is not None`, or use `.or_else(default)` / `.take()`. Returning bare `None` from a function typed `-> Int` won't compile — the reflex "return None on not-found" forces you to change the return type to `Optional[Int]`.

### No f-strings — template strings `t"..."`

`f"{x}"` doesn't exist. The note has used `"a " + name` throughout because that's the portable form; for interpolation, 1.0.0 has **template strings**:

```mojo
var count = 3
var msg = t"Give me {count} items."   # t"..." → a TString, lazy (no alloc yet)
print(msg)                            # materialises here
var s = String(msg)                   # or force it to a String explicitly
```

`t"..."` (and raw `rt"..."`) is a `TString` — the `{...}` fields take arbitrary expressions like Python's f-string, but the type is distinct and doesn't allocate until you print/convert it. Plain `+` concatenation and `String("x=", x)` also work.

### No `match` statement

Python 3.10+ `match`/`case` has no Mojo equivalent yet — the docs say so outright. Use `if`/`elif`. (Ternary `a if cond else b` and `for…else` / `while…else` *do* work.)

### `with` still works

RAII via `__deinit__` (§4) does *not* replace context managers — `with` is alive and behaves like Python, including multiple managers in one statement and custom `__enter__`/`__exit__` dunders:

```mojo
with open("f.txt", "r") as f:
    print(f.read())
```

### Closures and `lambda` — capture is explicit

Python closures capture surrounding names implicitly. Mojo makes the capture list explicit, in braces, and each captured name takes a convention (same vocabulary as arguments, §3):

```mojo
def apply_scaled(x: Int, factor: Int) -> Int:
    def scale(v: Int) {imm factor} -> Int:   # {imm factor}: capture factor by immutable ref
        return v * factor
    return scale(x)
```

Capture modes: bare name or `{imm x}` = immutable (default), `{var x}` = owned copy, `{mut x}` = mutable, `{var x^}` = moved in, `{ref x}` = caller-determined. There's also a typed `lambda` for one-off single-expression closures, using the same brace-capture syntax — handy for callbacks and comparators:

```mojo
var inc = lambda (x: Int) -> Int: x + 1     # body is a single expression
print(inc(41))                              # 42
```

A `lambda`/closure with no captures can be passed where a compile-time function pointer is expected; one that captures is a runtime value.

### `*args` / `**kwargs` have rules

They exist but aren't the free-for-all Python bags:

- `*args` is either **homogeneous** (`*vals: Int` → a `VariadicList`) or **heterogeneous** (a `VariadicPack` you iterate with `comptime for`).
- `**kwargs` must use the `var` convention, and **all values must share one type** that's `Copyable` (you read them as a `StringDict`). Dict-unpacking a call (`f(**d)`) is "not supported yet."
- Default values **can't** go on a `mut` argument. The `/` (positional-only) and `*` (keyword-only) markers work as in Python.

---

## 10. Tooling — Install, Run, Test, Ship

> *TL;DR — there's no `pip install mojo` and no `python foo.py`. You install with **pixi**, run with `mojo run` (JIT) or `mojo build` (native binary), test with a `TestSuite` (there is no `mojo test`), and split code into files where a directory needs `__init__.mojo` to be a package.*

The note's `mojo hello.mojo` only works *inside a project environment*. Here's the zero-to-running path a Python dev actually needs.

### Install — via `pixi`, not pip

Mojo 1.0.0 ships through **pixi** (Modular's chosen package manager) against a conda channel:

```bash
curl -fsSL https://pixi.sh/install.sh | sh          # one-time: install pixi
pixi init myproj -c https://conda.modular.com/max/ -c conda-forge
cd myproj
pixi add mojo                                        # adds the compiler to this project
pixi shell                                           # enter the env (or prefix commands with `pixi run`)
```

`pip install mojo` does not exist. Mojo is project-scoped, like a `venv` created for you — outside the pixi env the `mojo` command isn't on your PATH.

### The `mojo` CLI

```bash
mojo run app.mojo        # JIT-compile and execute (the everyday `python app.py`)
mojo build app.mojo      # AOT-compile to a standalone native binary → ./app
mojo repl                # interactive shell (the `python` REPL)
mojo format app.mojo     # autoformat (the black/ruff analogue)
mojo debug --vscode app.mojo   # launch under LLDB in VS Code
mojo doc / mojo demangle / mojo precompile   # docs, symbol demangling, package precompile
```

The `run` vs `build` split is the key idea: `run` is for iteration, `build` gives you a dependency-free binary — the payoff Python can't offer. Note what's **absent**: there is **no `mojo test`** and no `mojo package` subcommand (both existed in older tooling).

### Testing — `TestSuite`, run through `mojo run`

Since there's no `mojo test`, you assemble tests yourself and execute them like any program:

```mojo
from std.testing import assert_equal, TestSuite

def test_add():                # module-scope, name starts with test_, no args, raises on failure
    assert_equal(1 + 1, 2)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

Run with `mojo run test_math.mojo` (add `-I src` to point at your source tree); filter with `--only` / `--skip`.

### Splitting code across files — modules and packages

- A **module** is a single `.mojo` file; a sibling file can `from helpers import my_fn`.
- A **package** is a *directory*, and it **requires an `__init__.mojo`** — without that file Mojo won't treat the directory as importable (mirrors `__init__.py`, but the file is mandatory).
- **Unlike Python, top-level code doesn't run on import.** You can't put executable statements in `__init__.mojo` and have them fire when the package is imported — only declarations and re-exports.
- Importing across directories needs the search path flag: `mojo run -I src main.mojo`.

### Standard-library imports use the `std.` prefix

Built-ins (`print`, `List`, `Dict`, `String`, `Int`, `Optional` after import) need no import, but everything else is under `std.`:

```mojo
from std.python import Python
from std.math import sqrt
from std.collections import Dict
```

Pre-1.0 code wrote `from python import Python` / `from math import sqrt` — drop the `std.` and it won't resolve.

### Shipping a package

`mojo precompile mypkg -o mypkg.mojoc` produces a `.mojoc` archive — but it's **tied to the exact compiler version** that built it, so it's not a general-purpose distributable the way a wheel is. (Older tutorials say `mojo package` → `.mojopkg`; both are renamed.) For an application, prefer `mojo build` to a native binary.

### Editor

Modular ships an **official VS Code extension** (and Open-VSX builds) with completion, hover docs, formatting, and a language server — this is where deprecation fix-its (like `__del__` → `__deinit__`) surface. There's no first-party PyCharm/pylance path; VS Code is the supported story.

---

## Trade-offs

- **Python compatibility is partial.** Mojo accepts most Python *syntax* (def, indentation, control flow). It does not accept arbitrary Python *semantics* — class inheritance, dynamic dispatch, monkey-patching, descriptors, metaclasses, the GIL. Treat it as "starts looking like Python; quickly diverges."
- **`def` is statically typed.** Don't expect Python's dynamic-typing flexibility — every parameter needs a type annotation, return types are explicit, and errors don't propagate unless `raises` is on the signature. The Python *syntax* lets you start; the *semantics* are systems-language strict.
- **Ownership without lifetimes is friendlier than Rust — until it isn't.** No `<'a>` annotations is great. But when you eventually need parametric references (`ref [origin] T`), the model gets denser. Most Python users hit a ceiling much later than they'd hit Rust's.
- **Compile-time parameters are powerful and verbose.** `SIMD[DType.float32, 4]` is precise but noisy. Many functions end up with `[]` parameters that feel like ceremony until you start writing performance-critical code.
- **Ecosystem is small.** The 1.0.0 release (Aug 2026) stabilises the *language*, not the *ecosystem* — the package ecosystem is still young. You get NumPy/PyTorch via Python interop, but native Mojo libraries for web servers, async runtimes, CLI parsing, JSON, etc. are scarce. For non-AI work, Rust/Go/Python have much more.
- **Most docs and tutorials predate 1.0.0.** The stable release renamed a chunk of surface syntax (`fn`→`def`, `__del__`→`__deinit__`, `size`→`length`, the ownership keywords, `@value`, `alias`, `let`). Blog posts and older courses will not compile verbatim — budget time to translate them via the [renaming table](#older-syntax-youll-see-in-tutorials-renamed-or-removed).
- **Vendor lock-in concerns.** Mojo is open under the Apache 2.0 license but tightly bound to Modular's MAX platform for production deployment. The bet you make: Modular's stack will dominate AI inference. If it doesn't, your Mojo code becomes orphaned.

---

## Common Pitfalls

- **Using `let`** — it's gone. **Fix:** `var`.
- **Using `fn`** — deprecated as of 1.0.0. **Why:** `def` absorbed the strict, typed function role; `fn` is on its way out. **Fix:** write `def`.
- **Defining `__del__`** — the destructor dunder was renamed. **Why:** parity with `__init__`. **Fix:** `def __deinit__(deinit self):` (`__del__` still compiles but warns).
- **Passing `size=` to `SIMD`** — compile error since 1.0.0. **Why:** the parameter was renamed to `length`. **Fix:** use positional `SIMD[dt, N]` or `length=N`.
- **Writing `def main():` then calling Python without `raises`** — compile error. **Why:** Python interop can throw. **Fix:** `def main() raises:`.
- **Forgetting `^` when passing to a `var` parameter** — Mojo tries to copy; fails if the type isn't `Copyable`. **Fix:** explicit `value^` to transfer ownership.
- **Using a value after `^`** — uninitialised use. **Why:** transfer ends the source's lifetime. **Fix:** restructure so the producer is the last code that touches the value, or copy before transferring.
- **Mutating through a `def`-style parameter without `mut`** — silently operates on a copy or fails. **Fix:** declare `mut x:` in the function signature.
- **Reassigning a variable to a different type** — `x = 10; x = "foo"` fails. **Why:** types are statically frozen at first assignment. **Fix:** use a new binding name.
- **Mixing `[]` and `()` parameters wrongly** — `repeat[3](msg)` is correct; `repeat(3, msg)` won't typecheck. **Fix:** memorise that `[]` = compile-time, `()` = runtime.
- **Conforming a struct to `Copyable` without all fields being `Copyable`** — compile error. **Fix:** check the field types; non-copyable fields propagate.
- **Expecting Python's iterator protocol everywhere** — Mojo's `for x in container` works for things that implement the right iterator trait, not arbitrary `__iter__` like Python. **Fix:** check the trait requirement for the container you're iterating.
- **Treating Python interop as zero-cost** — every call crosses into CPython, which is expensive. **Fix:** batch your interop crossings (one big `np.dot` call beats a thousand `np.array` constructions).
- **Returning bare `None` from a typed function** — `-> Int` can't be `None`. **Why:** absence is its own type. **Fix:** return `Optional[Int]` and `return None` (§9).
- **Calling `.value()` on an empty `Optional`** — this *aborts the process*, it doesn't raise a catchable error. **Fix:** guard with `if opt:` / `opt is not None`, or use `.or_else(default)` (§9).
- **Writing an f-string `f"{x}"`** — not supported. **Fix:** template string `t"{x}"` or `String(...)` / `+` (§9).
- **`except SomeError as e:`** — Mojo has one `except` per `try` and no type matching. **Fix:** `except e:` (§9).
- **`var b = a` on a struct marked only `Copyable`** — compile error. **Why:** `Copyable` is *explicit* copy; only `ImplicitlyCopyable` copies on assignment. **Fix:** `var b = a.copy()` (§4).
- **Top-level `import numpy` / `Python.import_module` at module scope** — Mojo runs no top-level code. **Fix:** put the import inside `main()` or a helper (§8).
- **`pip install mojo` / `python app.mojo`** — neither exists. **Fix:** install via `pixi`, run with `mojo run app.mojo` inside the project env (§10).
- **Looking for `mojo test`** — there isn't one. **Fix:** write `test_*` functions and drive them with `TestSuite.discover_tests[...]().run()` via `mojo run` (§10).
- **Directory import fails silently** — a package folder needs an `__init__.mojo`, and top-level code in it does *not* run on import. **Fix:** add the file; keep it declarations/re-exports only (§10).

---

## Decision Table

| Scenario | Pick | Not | Why |
|---|---|---|---|
| Mixing the speed of compiled code with Python libraries | call NumPy/PyTorch via `Python.import_module`, write hot paths in pure Mojo | one or the other | Mojo's pitch is gradual migration |
| Library API meant for other developers | `def` with full type annotations and `raises` where needed | hand-waving the types | Contract is visible; better errors |
| Function takes a large value, reads only | default (borrowed) | `var` | No copy, no ownership transfer |
| Function mutates the caller's value | `mut x:` | `var x:` + return | Less ceremony; clear at call site |
| Function consumes the value (e.g., builder) | `var x:` + caller passes `value^` | `mut` | Explicit ownership transfer |
| Generic over types | `[T: Trait]` parameter | runtime dispatch | Compile-time specialisation |
| Generic over a numeric constant | `[N: Int]` parameter | runtime arg | Lets the compiler unroll/specialise |
| Vector arithmetic | `SIMD[dtype, width]` | `List[Float64]` | Single instruction per op |
| Need NumPy's API | Python interop | rewrite | Mojo can call NumPy directly |
| Need to hot-spot one Python function | Write Mojo extension, leave rest in Python | rewrite whole app | Gradual migration |
| Calling out to CPython many times in a tight loop | batch | per-iteration | CPython crossings are expensive |
| Struct that should auto-derive copy/move | `(Copyable)` trait conformance + `@fieldwise_init` | hand-rolled constructors | Cleaner, idiomatic |

---

## References

### Primary
- [Mojo Manual](https://mojolang.org/docs/manual/) — official, regularly updated (docs are versioned; this note tracks 1.0.0)
- [Mojo changelog / releases](https://mojolang.org/docs/changelog) — per-version change lists; 1.0.0 shipped 2026-08-11. **Watch for nightly-only entries** — the changelog mixes released and unreleased changes, so cross-check the versioned manual before trusting a claim
- [Mojo Language Basics](https://mojolang.org/docs/manual/basics) — syntax intro
- [Value ownership](https://mojolang.org/docs/manual/values/) — the `mut`/`var`/`ref` system
- [Value lifecycle](https://mojolang.org/docs/manual/lifecycle/) — constructors, destructors, traits
- [Metaprogramming](https://mojolang.org/docs/manual/metaprogramming/) — parameters and `comptime`
- [Traits](https://mojolang.org/docs/manual/traits) — definitions, conformance, built-ins
- [Structs](https://mojolang.org/docs/manual/structs) — defining types
- [Parameters](https://mojolang.org/docs/manual/parameters/) — compile-time parameter system
- [Python interop](https://mojolang.org/docs/manual/python/) — calling Python from Mojo and back
- [Python types & conversion](https://mojolang.org/docs/manual/python/types) — `PythonObject`, `py=` constructors, auto-conversion rules
- [GPU programming](https://mojolang.org/docs/manual/gpu/architecture/) — kernels and layouts
- [Error handling](https://mojolang.org/docs/manual/errors) — `try`/`except`, `raise`, context managers (§9)
- [Functions & closures](https://mojolang.org/docs/manual/functions) — args/kwargs, capture lists, `lambda` (§9)
- [Get started / install](https://mojolang.org/docs/manual/get-started) — pixi setup and first project (§10)
- [Mojo CLI](https://mojolang.org/docs/cli) — `run`/`build`/`repl`/`format`/`doc`/`debug` (§10)
- [Testing](https://mojolang.org/docs/tools/testing) — `TestSuite`, assert helpers (§10)
- [Packages & modules](https://mojolang.org/docs/manual/packages) — `__init__.mojo`, imports, `precompile` (§10)

### Background and context
- [Modular blog](https://www.modular.com/blog) — language announcements and design rationale
- [Mojo standard library source](https://github.com/modular/modular) — open-source stdlib for reference

### Community
- [Modular forum](https://forum.modular.com) — Q&A and announcements

## See Also

- [rust-fundamentals](../rust/rust-fundamentals.md) — Mojo's ownership (`mut`/`var`/`^`) is a friendlier variant of Rust's (`&mut`/`move`); Mojo's `[T: Trait]` parameters are roughly Rust's `<T: Trait>` generics; Mojo skips Rust's explicit lifetimes; Mojo's Python-shaped syntax has no Rust analogue.
- [rust-macros](../rust/rust-macros.md) — Mojo's compile-time parameter system overlaps with Rust's macro machinery, but Mojo's is *type-driven* (no token rewriting), making it closer to Rust's const generics than to `macro_rules!`.
- [golang-fundamentals](../golang/golang-fundamentals.md) — Mojo and Go both pitch ergonomic-vs-systems sweet spots; Go achieves it via simplicity and GC, Mojo via Python compatibility and explicit ownership. Different bets, different audiences.
