# Configuración del Compilador
CXX = g++
CXXFLAGS = -fPIC -shared -O3 -Wall
TARGET = ../libtentpu.so
SRC = src/engine.cpp

# Regla principal
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
	@echo "[✅] Motor Nativo PCURE-AI+ Compilado en $(TARGET)"

clean:
	rm -f $(TARGET)