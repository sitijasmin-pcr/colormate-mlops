"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

type AnalysisResponse = {
  status: string;
  message: string;
  image: {
    valid: boolean;
    format: string;
    width: number;
    height: number;
    size_bytes: number;
  };
  analysis: {
    predicted_class: string;
    season: string;
    confidence_score: number;
    class_index: number;
    model: string;
    experiment: string;
    checkpoint_epoch: number;
    device: string;
  };
  consent: {
    processing_consent: boolean;
    storage_consent: boolean;
    consent_version: string;
  };
  storage_status: string;
  disclaimer: string;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const MAX_FILE_SIZE = 5 * 1024 * 1024;

export default function ScanPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [photo, setPhoto] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [processingConsent, setProcessingConsent] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraActive(false);
  };

  const startCamera = async () => {
    setError("");
    setResult(null);
    setPhoto(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        "Browser tidak mendukung akses kamera atau koneksi belum aman. Gunakan localhost atau HTTPS."
      );
      return;
    }

    try {
      setLoading(true);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
    } catch (err) {
      console.error("Camera error:", err);
      setError(
        "Kamera tidak dapat diakses. Periksa izin kamera pada browser dan pastikan kamera tidak sedang digunakan aplikasi lain."
      );
    } finally {
      setLoading(false);
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;

    if (!video || !video.videoWidth) {
      setError("Kamera belum siap. Coba beberapa saat lagi.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
      setError("Tidak dapat memproses foto.");
      return;
    }

    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    setPhoto(canvas.toDataURL("image/jpeg", 0.9));
    setResult(null);
    setError("");
    stopCamera();
  };

  const handleUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (!file) return;

    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];

    if (!allowedTypes.includes(file.type)) {
      setError("Gunakan format JPG, PNG, atau WebP.");
      event.target.value = "";
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setError("Ukuran foto maksimal 5 MB.");
      event.target.value = "";
      return;
    }

    const reader = new FileReader();

    reader.onload = () => {
      if (typeof reader.result === "string") {
        setPhoto(reader.result);
        setResult(null);
        setError("");
        setProcessingConsent(false);
        stopCamera();
      }
    };

    reader.onerror = () => {
      setError("Foto gagal dibaca. Silakan coba kembali.");
    };

    reader.readAsDataURL(file);
    event.target.value = "";
  };

  const analyzePhoto = async () => {
    if (!photo) {
      setError("Silakan ambil atau upload foto terlebih dahulu.");
      return;
    }

    if (!processingConsent) {
      setError("Silakan setujui pemrosesan foto sebelum melanjutkan.");
      return;
    }

    try {
      setAnalyzing(true);
      setError("");
      setResult(null);

      const imageResponse = await fetch(photo);
      const imageBlob = await imageResponse.blob();

      const formData = new FormData();

      formData.append("file", imageBlob, "colormate-scan.jpg");
      formData.append("processing_consent", "true");
      formData.append("storage_consent", "false");
      formData.append("consent_version", "1.0");

      const response = await fetch(
        `${API_URL}/api/v1/analyze/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item: { msg?: string }) => item.msg).join(", ")
          : data.detail;

        throw new Error(
          detail || data.message || "Analisis foto gagal dilakukan."
        );
      }

      setResult(data as AnalysisResponse);
    } catch (err) {
      console.error("Analysis error:", err);

      setError(
        err instanceof Error
          ? err.message
          : "Terjadi kesalahan saat menghubungkan ke server."
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const resetPhoto = () => {
    setPhoto(null);
    setResult(null);
    setError("");
    setProcessingConsent(false);
  };

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const formatColorName = (value: string) =>
    value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());

  return (
    <main className="min-h-screen bg-[#fafbf7] text-[#253d32]">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <Link href="/" className="text-2xl font-bold tracking-tight">
          color<span className="text-[#82a78b]">mate.</span>
        </Link>

        <Link
          href="/"
          className="text-sm font-medium text-[#718071] hover:text-[#527d60]"
        >
          ← Back to Home
        </Link>
      </nav>

      <section className="mx-auto max-w-5xl px-5 pb-20 pt-8">
        <div className="mx-auto max-w-2xl text-center">
          <span className="rounded-full bg-[#eef4ec] px-4 py-2 text-xs font-semibold tracking-widest text-[#527d60]">
            PERSONAL COLOR ANALYSIS
          </span>

          <h1 className="mt-7 font-serif text-4xl md:text-5xl">
            Let's discover your{" "}
            <span className="italic text-[#82a78b]">colors.</span>
          </h1>

          <p className="mx-auto mt-5 max-w-xl text-sm leading-7 text-[#849084]">
            Ambil foto wajahmu untuk memulai perjalanan menemukan warna
            makeup yang paling sesuai dengan karakteristik wajahmu.
          </p>
        </div>

        <div className="mx-auto mt-12 grid max-w-4xl gap-8 md:grid-cols-[1.3fr_0.7fr]">
          {/* PHOTO PREVIEW */}
          <div className="rounded-[2rem] border border-[#e6ecdf] bg-white p-4 shadow-lg shadow-[#526c52]/5">
            <div className="relative flex min-h-[350px] items-center justify-center overflow-hidden rounded-[1.5rem] bg-[#e9eee4] md:min-h-[430px]">
              {photo ? (
                <img
                  src={photo}
                  alt="Preview foto wajah"
                  className="h-full max-h-[520px] w-full object-contain"
                />
              ) : cameraActive ? (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="h-full min-h-[350px] w-full scale-x-[-1] object-cover md:min-h-[430px]"
                />
              ) : (
                <div className="px-8 text-center">
                  <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-full bg-white text-4xl text-[#82a78b]">
                    ♡
                  </div>

                  <h2 className="mt-6 text-lg font-semibold">
                    Your camera preview
                  </h2>

                  <p className="mt-2 text-sm leading-6 text-[#879187]">
                    Aktifkan kamera atau upload foto untuk memulai analisis.
                  </p>
                </div>
              )}

              {cameraActive && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                  <div className="h-[75%] w-[65%] rounded-[48%] border-2 border-dashed border-white/80" />
                </div>
              )}
            </div>

            {error && (
              <div
                role="alert"
                className="mt-4 rounded-xl bg-red-50 p-3 text-sm leading-6 text-red-700"
              >
                {error}
              </div>
            )}

            {/* CAMERA AND UPLOAD BUTTONS */}
            <div className="mt-5 flex flex-wrap justify-center gap-3">
              {photo ? (
                <button
                  onClick={resetPhoto}
                  disabled={analyzing}
                  className="rounded-full border border-[#dce5d9] px-7 py-3 text-sm font-semibold hover:bg-[#f5f7f1] disabled:opacity-50"
                >
                  Retake Photo
                </button>
              ) : cameraActive ? (
                <>
                  <button
                    onClick={capturePhoto}
                    className="rounded-full bg-[#527d60] px-8 py-3 text-sm font-semibold text-white shadow-lg shadow-[#527d60]/20 hover:bg-[#3f674d]"
                  >
                    Capture Photo
                  </button>

                  <button
                    onClick={stopCamera}
                    className="rounded-full border border-[#dce5d9] px-7 py-3 text-sm font-semibold hover:bg-[#f5f7f1]"
                  >
                    Stop Camera
                  </button>
                </>
              ) : (
                <button
                  onClick={startCamera}
                  disabled={loading}
                  className="rounded-full bg-[#527d60] px-8 py-3 text-sm font-semibold text-white shadow-lg shadow-[#527d60]/20 hover:bg-[#3f674d] disabled:opacity-50"
                >
                  {loading ? "Opening Camera..." : "Open Camera"}
                </button>
              )}

              <label
                className={`cursor-pointer rounded-full border border-[#dce5d9] px-7 py-3 text-sm font-semibold hover:bg-[#f5f7f1] ${
                  analyzing ? "pointer-events-none opacity-50" : ""
                }`}
              >
                Upload Photo
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  capture="user"
                  disabled={analyzing}
                  className="hidden"
                  onChange={handleUpload}
                />
              </label>
            </div>
          </div>

          {/* INFORMATION AND ANALYSIS */}
          <aside className="space-y-5">
            <div className="rounded-3xl border border-[#e6ecdf] bg-white p-6">
              <h2 className="font-semibold">Before you scan</h2>

              <ul className="mt-5 space-y-4 text-sm leading-6 text-[#7d897d]">
                <li>☀️ Gunakan pencahayaan alami yang cukup.</li>
                <li>📸 Hadapkan wajah langsung ke kamera.</li>
                <li>💄 Hindari makeup yang terlalu tebal.</li>
                <li>🎨 Hindari filter dan efek kamera.</li>
                <li>🧕 Pastikan wajah terlihat jelas.</li>
              </ul>
            </div>

            <div className="rounded-3xl bg-[#edf3e9] p-6">
              <div className="text-sm font-semibold text-[#527d60]">
                Your privacy matters
              </div>

              <p className="mt-3 text-xs leading-6 text-[#7d897d]">
                Foto dikirim ke server untuk proses analisis dan tidak
                disimpan secara permanen oleh fitur ini. Pemrosesan hanya
                dilakukan setelah kamu memberikan persetujuan.
              </p>
            </div>

            {/* PHOTO CONSENT AND ANALYSIS BUTTON */}
            {photo && !result && (
              <div className="rounded-3xl border border-[#dce8dc] bg-white p-6">
                <div className="font-semibold text-[#527d60]">
                  ✓ Photo ready!
                </div>

                <p className="mt-2 text-xs leading-6 text-[#849084]">
                  Foto berhasil disiapkan. Berikan persetujuan untuk
                  mengirim foto ke sistem Computer Vision ColorMate.
                </p>

                <label className="mt-5 flex cursor-pointer items-start gap-3 text-xs leading-5 text-[#718071]">
                  <input
                    type="checkbox"
                    checked={processingConsent}
                    onChange={(event) =>
                      setProcessingConsent(event.target.checked)
                    }
                    disabled={analyzing}
                    className="mt-1 h-4 w-4 accent-[#527d60]"
                  />

                  <span>
                    Saya menyetujui foto ini diproses oleh sistem AI
                    ColorMate untuk keperluan analisis personal color.
                    Foto tidak dimaksudkan untuk disimpan secara permanen.
                  </span>
                </label>

                <button
                  onClick={analyzePhoto}
                  disabled={!processingConsent || analyzing}
                  className="mt-5 w-full rounded-full bg-[#527d60] px-5 py-3 text-sm font-semibold text-white hover:bg-[#3f674d] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {analyzing ? "Analyzing Your Colors..." : "Analyze My Colors"}
                </button>

                {analyzing && (
                  <p className="mt-3 text-center text-xs text-[#849084]">
                    Please wait while ColorMate analyzes your photo...
                  </p>
                )}
              </div>
            )}

            {/* ANALYSIS RESULT */}
            {result && (
              <div className="rounded-3xl border border-[#dce8dc] bg-white p-6">
                <span className="text-xs font-semibold tracking-widest text-[#82a78b]">
                  YOUR COLOR ANALYSIS
                </span>

                <h2 className="mt-3 font-serif text-3xl text-[#527d60]">
                  {formatColorName(result.analysis.predicted_class)}
                </h2>

                <p className="mt-2 text-sm text-[#849084]">
                  {result.message}
                </p>

                <div className="mt-6">
                  <div className="flex justify-between text-sm">
                    <span className="text-[#849084]">AI confidence</span>
                    <strong className="text-[#527d60]">
                      {(result.analysis.confidence_score * 100).toFixed(2)}%
                    </strong>
                  </div>

                  <div
                    className="mt-2 h-2 overflow-hidden rounded-full bg-[#edf3e9]"
                    role="progressbar"
                    aria-valuenow={result.analysis.confidence_score * 100}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label="AI confidence"
                  >
                    <div
                      className="h-full rounded-full bg-[#82a78b] transition-all duration-500"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(
                            0,
                            result.analysis.confidence_score * 100
                          )
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="mt-5 space-y-2 rounded-xl bg-[#f5f7f1] p-4 text-xs text-[#718071]">
                  <p>
                    <strong>Model:</strong> {result.analysis.model}
                  </p>
                  <p>
                    <strong>Experiment:</strong> {result.analysis.experiment}
                  </p>
                  <p>
                    <strong>Checkpoint:</strong> Epoch{" "}
                    {result.analysis.checkpoint_epoch}
                  </p>
                  <p>
                    <strong>Processing device:</strong>{" "}
                    {result.analysis.device}
                  </p>
                  <p>
                    <strong>Image:</strong> {result.image.width} ×{" "}
                    {result.image.height} ({result.image.format})
                  </p>
                  <p>
                    <strong>Photo storage:</strong>{" "}
                    {result.storage_status}
                  </p>
                </div>

                <div className="mt-5 rounded-xl border border-[#e6ecdf] p-4">
                  <p className="text-xs font-semibold text-[#527d60]">
                    Important note
                  </p>

                  <p className="mt-2 text-xs leading-6 text-[#849084]">
                    {result.disclaimer}
                  </p>

                  <p className="mt-2 text-xs leading-5 text-[#849084]">
                    Confidence is the model's prediction score, not a
                    measure of how accurate the personal color assessment
                    is.
                  </p>
                </div>

                <button
                  onClick={resetPhoto}
                  className="mt-5 w-full rounded-full border border-[#dce5d9] px-5 py-3 text-sm font-semibold text-[#527d60] hover:bg-[#f5f7f1]"
                >
                  Analyze Another Photo
                </button>
              </div>
            )}
          </aside>
        </div>
      </section>
    </main>
  );
}