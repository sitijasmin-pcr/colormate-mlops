const features = [
  {
    number: "01",
    title: "Personal Color Analysis",
    description:
      "Kenali karakter warna yang paling sesuai dengan warna alami wajahmu.",
    icon: "◉",
  },
  {
    number: "02",
    title: "Makeup Recommendation",
    description:
      "Temukan pilihan warna dan produk makeup berdasarkan hasil analisismu.",
    icon: "✧",
  },
  {
    number: "03",
    title: "AI Beauty Assistant",
    description:
      "Diskusikan hasil personal color dan temukan jawaban melalui asisten AI.",
    icon: "✳",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-[#fafbf7] text-[#253d32]">
      {/* Navbar */}
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6 lg:px-12">
        <a href="#" className="text-2xl font-bold tracking-tight">
          color<span className="text-[#82a78b]">mate.</span>
        </a>

        <div className="hidden items-center gap-9 text-sm font-medium md:flex">
          <a href="#features" className="hover:text-[#82a78b]">
            Features
          </a>
          <a href="#about" className="hover:text-[#82a78b]">
            About
          </a>
          <a href="#how-it-works" className="hover:text-[#82a78b]">
            How It Works
          </a>
        </div>

        <a
          href="#start"
          className="rounded-full bg-[#527d60] px-6 py-3 text-sm font-semibold text-white transition hover:bg-[#3f674d]"
        >
          Get Started
        </a>
      </nav>

      {/* Hero */}
      <section
        id="about"
        className="mx-auto grid max-w-7xl items-center gap-12 px-6 pb-24 pt-12 lg:grid-cols-2 lg:px-12 lg:pb-32 lg:pt-20"
      >
        <div>
          <span className="mb-7 inline-flex rounded-full border border-[#dce8dc] bg-[#eef4ec] px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-[#527d60]">
            Your Beauty, Your Colors
          </span>

          <h1 className="max-w-xl text-5xl leading-[1.12] font-semibold tracking-tight md:text-7xl">
            Discover the
            <span className="block font-serif font-normal italic text-[#789b7d]">
              Colors
            </span>
            That Define You.
          </h1>

          <p className="mt-7 max-w-lg text-base leading-8 text-[#78877b]">
            Temukan personal color-mu, eksplorasi warna makeup yang sesuai,
            dan kenali kecantikan unikmu bersama ColorMate.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-4">
            <a
              id="start"
              href="#how-it-works"
              className="rounded-full bg-[#527d60] px-8 py-4 text-sm font-semibold text-white shadow-lg shadow-[#527d60]/15 transition hover:-translate-y-1 hover:bg-[#3f674d]"
            >
              Start Your Analysis →
            </a>
            <a
              href="#features"
              className="rounded-full border border-[#dce5d9] px-8 py-4 text-sm font-semibold transition hover:bg-[#eef4ec]"
            >
              Explore Features
            </a>
          </div>

          <div className="mt-12 flex items-center gap-4">
            <div className="flex -space-x-2">
              {["#c9a58e", "#e2c3a6", "#a87862", "#d6aa91"].map(
                (color, index) => (
                  <div
                    key={index}
                    className="h-9 w-9 rounded-full border-2 border-white"
                    style={{ backgroundColor: color }}
                  />
                )
              )}
            </div>
            <p className="text-xs text-[#879187]">
              Discover your unique beauty
            </p>
          </div>
        </div>

        {/* Visual Preview */}
        <div className="relative mx-auto w-full max-w-lg">
          <div className="absolute -inset-5 rounded-[45%] bg-[#e7eee1] blur-2xl" />

          <div className="relative overflow-hidden rounded-[2rem] border border-white bg-[#e9eee4] p-5 shadow-xl shadow-[#526c52]/10">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold">Your Color Journey</span>
              <span className="rounded-full bg-white/80 px-3 py-1 text-xs text-[#527d60]">
                COLOR ANALYSIS
              </span>
            </div>

            <div className="mt-5 flex min-h-[350px] items-center justify-center rounded-2xl bg-gradient-to-br from-[#f6e9db] via-[#e7d4c0] to-[#d6c6ad]">
              <div className="relative flex h-64 w-52 items-center justify-center rounded-[48%] bg-[#c69d7f] shadow-inner">
                <div className="absolute top-20 left-10 h-2 w-3 rounded-full bg-[#624c3d]" />
                <div className="absolute top-20 right-10 h-2 w-3 rounded-full bg-[#624c3d]" />
                <div className="absolute top-32 h-7 w-10 rounded-full border-b-4 border-[#8d504c]" />
                <div className="absolute -bottom-3 h-24 w-64 rounded-t-[50%] bg-[#527d60]" />
              </div>
            </div>

            <div className="mt-5 rounded-2xl bg-white p-5">
              <p className="text-xs tracking-widest text-[#8b9a89]">
                YOUR PERSONAL COLOR
              </p>
              <div className="mt-2 flex items-center justify-between">
                <span className="font-serif text-2xl italic">Discover yours</span>
                <span className="text-xl text-[#82a78b]">✧</span>
              </div>
              <div className="mt-4 flex gap-2">
                {["#d7b89b", "#b58c79", "#a9b69a", "#d3c1aa", "#879d84"].map(
                  (color) => (
                    <div
                      key={color}
                      className="h-9 flex-1 rounded-lg"
                      style={{ backgroundColor: color }}
                    />
                  )
                )}
              </div>
            </div>
          </div>
          <div className="absolute -right-4 top-24 rounded-xl bg-white px-4 py-3 text-xs font-medium shadow-lg">
            ✨ Your beauty, your palette
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="bg-white px-6 py-24 lg:px-12">
        <div className="mx-auto max-w-7xl">
          <div className="mx-auto max-w-xl text-center">
            <p className="text-xs font-semibold tracking-[0.2em] text-[#82a78b]">
              WHAT COLORMATE OFFERS
            </p>
            <h2 className="mt-4 font-serif text-4xl">
              Beauty begins with understanding your colors.
            </h2>
            <p className="mt-4 text-sm leading-7 text-[#879187]">
              Satu pengalaman untuk mengenal warna, memilih makeup, dan
              menemukan inspirasi kecantikanmu.
            </p>
          </div>

          <div className="mt-14 grid gap-6 md:grid-cols-3">
            {features.map((feature) => (
              <div
                key={feature.number}
                className="rounded-3xl border border-[#edf0e9] bg-[#fafbf7] p-8 transition hover:-translate-y-1 hover:shadow-lg"
              >
                <div className="flex items-center justify-between">
                  <span className="text-3xl text-[#82a78b]">
                    {feature.icon}
                  </span>
                  <span className="text-xs text-[#a5afa1]">
                    {feature.number}
                  </span>
                </div>
                <h3 className="mt-8 text-lg font-semibold">
                  {feature.title}
                </h3>
                <p className="mt-3 text-sm leading-7 text-[#879187]">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="px-6 py-24 lg:px-12">
        <div className="mx-auto max-w-7xl text-center">
          <p className="text-xs font-semibold tracking-[0.2em] text-[#82a78b]">
            SIMPLE & PERSONAL
          </p>
          <h2 className="mt-4 font-serif text-4xl">
            Your journey starts here.
          </h2>

          <div className="mt-14 grid gap-8 text-left md:grid-cols-3">
            {[
              ["01", "Scan Your Face", "Gunakan kamera HP untuk mengambil foto wajah."],
              ["02", "Discover Your Color", "Dapatkan hasil analisis personal color."],
              ["03", "Explore Your Beauty", "Lihat rekomendasi makeup dan konsultasikan dengan AI."],
            ].map(([number, title, description]) => (
              <div key={number} className="border-t border-[#dce5d9] pt-6">
                <span className="text-sm text-[#82a78b]">{number}</span>
                <h3 className="mt-4 text-xl font-semibold">{title}</h3>
                <p className="mt-3 text-sm leading-7 text-[#879187]">
                  {description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 pb-20 lg:px-12">
        <div className="mx-auto max-w-7xl rounded-[2rem] bg-[#527d60] px-8 py-16 text-center text-white md:px-16">
          <p className="text-xs tracking-[0.2em] text-[#d7e6d5]">
            MEET YOUR COLORS
          </p>
          <h2 className="mx-auto mt-5 max-w-2xl font-serif text-4xl leading-tight md:text-5xl">
            Your beauty is unique. Let your colors tell the story.
          </h2>
          <a
            href="#start"
            className="mt-8 inline-block rounded-full bg-white px-8 py-4 text-sm font-semibold text-[#527d60] transition hover:bg-[#edf4e9]"
          >
            Discover Your Colors →
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-[#e8ede4] bg-white px-6 py-8 lg:px-12">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 text-sm text-[#879187] md:flex-row">
          <span className="text-xl font-bold text-[#253d32]">
            color<span className="text-[#82a78b]">mate.</span>
          </span>
          <span>Discover your colors. Celebrate your beauty.</span>
          <span>© 2026 ColorMate</span>
        </div>
      </footer>
    </main>
  );
}