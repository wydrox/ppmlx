class Ppmlx < Formula
  include Language::Python::Virtualenv

  desc "CLI for running LLMs on Apple Silicon via MLX"
  homepage "https://ppmlx.dev"
  url "https://files.pythonhosted.org/packages/ee/0a/433431922f521f2bab089399880f59768cc3b1ecc4d53848be287bfdb26b/ppmlx-0.1.0.tar.gz"
  sha256 "22f94d51c01930f8f2dd865bca022cbdb7711afc70fdc87e663b61121edbecff"
  license "MIT"
  head "https://github.com/wydrox/ppmlx.git", branch: "main"

  depends_on "python@3.11"
  depends_on :macos
  depends_on arch: :arm64

  def install
    virtualenv_create(libexec, "python3.11")

    # Install ppmlx with all dependencies (including optional embeddings)
    # in a single pip invocation. This lets pip handle dependency resolution
    # and avoids maintaining a separate list that drifts from pyproject.toml.
    system libexec/"bin/pip", "install", ".[embeddings]"

    (bin/"ppmlx").write_env_script libexec/"bin/ppmlx", PATH: "#{libexec}/bin:#{ENV["PATH"]}"
  end

  def caveats
    <<~EOS
      ppmlx requires Apple Silicon (M1/M2/M3/M4) and macOS 13+.

      Quick start:
        ppmlx pull llama3
        ppmlx run llama3
        ppmlx serve          # OpenAI-compatible API on :6767
    EOS
  end

  test do
    assert_match "ppmlx", shell_output("#{bin}/ppmlx --help")
  end
end
