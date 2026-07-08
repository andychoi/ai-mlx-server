class AiMlxServer < Formula
  include Language::Python::Virtualenv

  desc "MLX inference server for Apple Silicon — OpenAI and Ollama compatible API"
  homepage "https://github.com/andychoi/mlx-server"
  url "https://github.com/andychoi/mlx-server/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"  # Update when tagging
  license "MIT"
  head "https://github.com/andychoi/mlx-server.git", branch: "main"

  depends_on "python@3.11"
  depends_on :macos
  # MLX requires Apple Silicon
  on_arm do
    # mlx only runs on Apple Silicon
  end

  def install
    virtualenv_install_with_resources

    # Install the launchd service installer script
    bin.install "packaging/install-service.sh" => "mlx-server-install-service"
  end

  def caveats
    <<~EOS
      mlx-server is installed. To run it:
        mlx-server --port 11434

      To install as a macOS background service:
        mlx-server-install-service
        launchctl load ~/Library/LaunchAgents/com.andychoi.mlx-server.plist

      Configuration: ~/.config/mlx-server/models.yaml
      Logs:          ~/Library/Logs/mlx-server.log

      Note: This requires Apple Silicon (MLX does not run on Intel Macs).
    EOS
  end

  test do
    system "#{bin}/mlx-server", "--help"
  end
end
