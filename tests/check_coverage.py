import sys
import xml.etree.ElementTree as ET

COVERAGE_XML = "reports/coverage.xml"
REQUIRED_COVERAGE = 100.0

def main():
    tree = ET.parse(COVERAGE_XML)
    root = tree.getroot()
    coverage = float(root.attrib["line-rate"]) * 100
    print(f"Total coverage: {coverage:.2f}%")
    if coverage < REQUIRED_COVERAGE:
        print(f"::error::Test coverage {coverage:.2f}% is below required {REQUIRED_COVERAGE}%")
        sys.exit(1)

if __name__ == "__main__":
    main()
