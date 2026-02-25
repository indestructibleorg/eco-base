#!/bin/bash

# Multi-language build script for CodeQL analysis
# Usage: ./build-matrix.sh <language> [build_mode]

LANGUAGE=$1
BUILD_MODE=${2:-"standard"}

echo "=========================================="
echo "Building for language: $LANGUAGE"
echo "Build mode: $BUILD_MODE"
echo "=========================================="

case "$LANGUAGE" in
  "java")
    echo "Building Java project..."
    if [ -f "build.gradle" ] || [ -f "build.gradle.kts" ]; then
      echo "Using Gradle..."
      if [ ! -x "./gradlew" ]; then
        echo "Making gradlew executable..."
        chmod +x ./gradlew
      fi
      ./gradlew clean build -x test --no-daemon
    elif [ -f "pom.xml" ]; then
      echo "Using Maven..."
      mvn clean compile -DskipTests
    else
      echo "No Java build file found (build.gradle or pom.xml)"
    fi
    ;;
  
  "javascript" | "typescript")
    echo "Building JavaScript/TypeScript project..."
    if [ -f "package-lock.json" ]; then
      echo "Using npm with package-lock.json..."
      npm ci --production=false
    elif [ -f "yarn.lock" ]; then
      echo "Using Yarn..."
      yarn install --frozen-lockfile
    elif [ -f "pnpm-lock.yaml" ]; then
      echo "Using pnpm..."
      pnpm install --frozen-lockfile
    elif [ -f "package.json" ]; then
      echo "Using npm without lockfile..."
      npm install
    fi
    
    # Build if build script exists
    if [ -f "package.json" ]; then
      if grep -q '"build"' package.json; then
        echo "Running build script..."
        npm run build
      fi
    fi
    ;;
  
  "python")
    echo "Setting up Python environment..."
    python -m pip install --upgrade pip
    
    if [ -f "requirements.txt" ]; then
      echo "Installing requirements.txt..."
      pip install -r requirements.txt
    fi
    
    if [ -f "setup.py" ]; then
      echo "Installing package in development mode..."
      pip install -e .
    fi
    
    if [ -f "pyproject.toml" ]; then
      echo "Installing with pyproject.toml..."
      pip install -e .
    fi
    ;;
  
  "csharp")
    echo "Building C# project..."
    if [ -f "*.sln" ] || [ -f "*.csproj" ]; then
      echo "Restoring NuGet packages..."
      dotnet restore
      echo "Building in Release configuration..."
      dotnet build --configuration Release --no-restore
    else
      echo "No C# project file found"
    fi
    ;;
  
  "go")
    echo "Building Go project..."
    if [ -f "go.mod" ]; then
      echo "Downloading Go modules..."
      go mod download
      echo "Verifying Go modules..."
      go mod verify
      echo "Building Go packages..."
      go build ./...
    else
      echo "No go.mod found"
    fi
    ;;
  
  "cpp" | "c")
    echo "Building C/C++ project..."
    if [ -f "CMakeLists.txt" ]; then
      echo "Using CMake..."
      mkdir -p build && cd build
      cmake .. -DCMAKE_BUILD_TYPE=Release
      make -j$(nproc)
    elif [ -f "Makefile" ]; then
      echo "Using Makefile..."
      make clean && make release
    else
      echo "No C/C++ build file found (CMakeLists.txt or Makefile)"
    fi
    ;;
  
  "ruby")
    echo "Setting up Ruby environment..."
    if [ -f "Gemfile" ]; then
      echo "Installing gems..."
      bundle install
    fi
    ;;
  
  "swift")
    echo "Building Swift project..."
    if [ -f "Package.swift" ]; then
      echo "Building Swift package..."
      swift build -c release
    elif [ -f "*.xcodeproj" ] || [ -f "*.xcworkspace" ]; then
      echo "Building Xcode project..."
      xcodebuild -scheme "${XCODE_SCHEME:-$(xcodebuild -list | grep -A 1 Schemes | tail -1 | xargs)}" -configuration Release clean build
    fi
    ;;
  
  *)
    echo "Unknown language: $LANGUAGE"
    echo "Supported languages: java, javascript, typescript, python, csharp, go, cpp, c, ruby, swift"
    exit 1
    ;;
esac

BUILD_EXIT_CODE=$?

echo "=========================================="
if [ $BUILD_EXIT_CODE -eq 0 ]; then
  echo "✅ Build completed successfully"
else
  echo "❌ Build failed with exit code: $BUILD_EXIT_CODE"
fi
echo "=========================================="

exit $BUILD_EXIT_CODE
