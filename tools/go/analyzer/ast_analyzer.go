package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/tools/go/packages"
)

type FunctionInfo struct {
	Name         string   `json:"name"`
	ReceiverType string   `json:"receiver_type,omitempty"`
	Package      string   `json:"package"`
	File         string   `json:"file"`
	Line         int      `json:"line"`
	DocString    string   `json:"docstring,omitempty"`
	Calls        []string `json:"calls"`
	Context      string   `json:"context"`
}

type StructInfo struct {
	Name      string         `json:"name"`
	Package   string         `json:"package"`
	File      string         `json:"file"`
	Line      int            `json:"line"`
	DocString string         `json:"docstring,omitempty"`
	Fields    []FieldInfo    `json:"fields"`
	Methods   []FunctionInfo `json:"methods"`
	Context   string         `json:"context"`
}

type FieldInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
	Tag  string `json:"tag,omitempty"`
}

type InterfaceInfo struct {
	Name      string   `json:"name"`
	Package   string   `json:"package"`
	File      string   `json:"file"`
	Line      int      `json:"line"`
	DocString string   `json:"docstring,omitempty"`
	Methods   []string `json:"methods"`
	Context   string   `json:"context"`
}

type PackageInfo struct {
	Name       string          `json:"name"`
	Path       string          `json:"path"`
	Files      []string        `json:"files"`
	Imports    []string        `json:"imports"`
	Functions  []FunctionInfo  `json:"functions"`
	Structs    []StructInfo    `json:"structs"`
	Interfaces []InterfaceInfo `json:"interfaces"`
}

func main() {
	pkgPath := flag.String("pkg", ".", "package path to analyze")
	dirPath := flag.String("dir", "", "directory path to analyze")
	outputFile := flag.String("output", "", "output JSON file")
	specificPackage := flag.String("package", "", "specific package name to analyze (e.g. 'main')")
	excludeTests := flag.Bool("exclude-tests", false, "exclude test files (files ending with _test.go)")
	flag.Parse()

	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedSyntax |
			packages.NeedTypes | packages.NeedTypesInfo | packages.NeedDeps,
		Tests: !*excludeTests,
	}

	var pkgs []*packages.Package
	var err error

	// If a directory path is provided, use it for analysis
	if *dirPath != "" {
		// Set the working directory to the specified directory
		oldDir, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting current directory: %v\n", err)
			os.Exit(1)
		}

		err = os.Chdir(*dirPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error changing to directory %s: %v\n", *dirPath, err)
			os.Exit(1)
		}

		// Analyze all Go files in the directory
		pkgs, err = packages.Load(cfg, "./...")

		// Restore the original working directory
		_ = os.Chdir(oldDir)
	} else {
		// Use the package path for analysis
		pkgs, err = packages.Load(cfg, *pkgPath)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading package: %v\n", err)
		// Try direct file processing instead of exiting immediately
	}

	// Check if we need to process files directly (if package loading failed or returned empty)
	if len(pkgs) == 0 || (len(pkgs) > 0 && len(pkgs[0].Syntax) == 0) {
		// Try to process files directly
		result := processDirectFiles(*dirPath, *specificPackage, *excludeTests)

		// Output the result
		output, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(output))
		return
	}

	result := make(map[string]PackageInfo)

	for _, pkg := range pkgs {
		// Skip if we're looking for a specific package and this isn't it
		if *specificPackage != "" && pkg.Name != *specificPackage {
			fmt.Println("Ignoring package", pkg.Name)
			continue
		}

		if pkg.Errors != nil {
			fmt.Fprintf(os.Stderr, "Package %s has errors: %v\n", pkg.PkgPath, pkg.Errors)
			// Continue anyway with as much information as we have
		}

		pkgInfo := PackageInfo{
			Name:  pkg.Name,
			Path:  pkg.PkgPath,
			Files: pkg.GoFiles,
		}

		// Filter out test files if requested
		if *excludeTests {
			var filteredFiles []string
			for _, file := range pkgInfo.Files {
				if !strings.HasSuffix(file, "_test.go") {
					filteredFiles = append(filteredFiles, file)
				}
			}
			pkgInfo.Files = filteredFiles
		}

		// Get imports
		for _, f := range pkg.Syntax {
			// Skip test files if requested
			filename := pkg.Fset.Position(f.Pos()).Filename
			if *excludeTests && strings.HasSuffix(filename, "_test.go") {
				continue
			}

			for _, imp := range f.Imports {
				impPath := strings.Trim(imp.Path.Value, "\"")
				if !contains(pkgInfo.Imports, impPath) {
					pkgInfo.Imports = append(pkgInfo.Imports, impPath)
				}
			}
		}

		// Map to track method receivers for organizing methods with their structs
		structMethods := make(map[string][]FunctionInfo)

		// Analyze package contents
		for _, f := range pkg.Syntax {
			// Skip test files if requested
			filename := pkg.Fset.Position(f.Pos()).Filename
			if *excludeTests && strings.HasSuffix(filename, "_test.go") {
				continue
			}

			ast.Inspect(f, func(n ast.Node) bool {
				switch node := n.(type) {
				case *ast.FuncDecl:
					// Get function info
					fInfo := analyzeFuncDecl(node, pkg, f)
					pkgInfo.Functions = append(pkgInfo.Functions, fInfo)

					// If this is a method, record it for the struct
					if node.Recv != nil && len(node.Recv.List) > 0 {
						var receiverType string
						switch t := node.Recv.List[0].Type.(type) {
						case *ast.StarExpr:
							if ident, ok := t.X.(*ast.Ident); ok {
								receiverType = ident.Name
							}
						case *ast.Ident:
							receiverType = t.Name
						}

						if receiverType != "" {
							structMethods[receiverType] = append(structMethods[receiverType], fInfo)
						}
					}

				case *ast.TypeSpec:
					// Get type info (struct or interface)
					switch t := node.Type.(type) {
					case *ast.StructType:
						sInfo := analyzeStructType(node, t, pkg, f)
						pkgInfo.Structs = append(pkgInfo.Structs, sInfo)
					case *ast.InterfaceType:
						iInfo := analyzeInterfaceType(node, t, pkg, f)
						pkgInfo.Interfaces = append(pkgInfo.Interfaces, iInfo)
					}
				}
				return true
			})
		}

		// Associate methods with structs
		for i, structInfo := range pkgInfo.Structs {
			if methods, ok := structMethods[structInfo.Name]; ok {
				pkgInfo.Structs[i].Methods = methods
			}
		}

		result[pkg.PkgPath] = pkgInfo
	}

	// Ensure we have at least one package in the result
	if len(result) == 0 {
		// Process files directly as a fallback
		result = processDirectFiles(*dirPath, *specificPackage, *excludeTests)
	}

	// Output results
	if *outputFile != "" {
		output, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error marshaling JSON: %v\n", err)
			os.Exit(1)
		}
		err = os.WriteFile(*outputFile, output, 0644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error writing output file: %v\n", err)
			os.Exit(1)
		}
	} else {
		output, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(output))
	}
}

// Function to process files directly, used when package loading fails
func processDirectFiles(dirPath string, specificPackage string, excludeTests bool) map[string]PackageInfo {
	result := make(map[string]PackageInfo)

	// Default package info
	pkgInfo := PackageInfo{
		Name: "main", // Assume main package for simplicity
		Path: dirPath,
	}

	// Find .go files
	var goFiles []string
	if dirPath != "" {
		entries, err := os.ReadDir(dirPath)
		if err == nil {
			for _, entry := range entries {
				if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go") {
					// Skip test files if requested
					if excludeTests && strings.HasSuffix(entry.Name(), "_test.go") {
						continue
					}

					filePath := filepath.Join(dirPath, entry.Name())
					goFiles = append(goFiles, filePath)

					// Process this file to get the package name
					// Check for nil or empty specific package
					if specificPackage == "" || (pkgInfo.Name != "" && pkgInfo.Name == specificPackage) {
						processGoFile(filePath, &pkgInfo)
					}
				}
			}
		}
	}

	// If a specific package is requested but not found, return empty result
	if specificPackage != "" && pkgInfo.Name != specificPackage {
		return result
	}

	pkgInfo.Files = goFiles

	// If dirPath is empty, use a default key
	key := dirPath
	if key == "" {
		key = "."
	}

	result[key] = pkgInfo
	return result
}

// Process a single Go file and update package info
func processGoFile(filePath string, pkgInfo *PackageInfo) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing file %s: %v\n", filePath, err)
		return
	}

	// Update package name if it was unknown
	if pkgInfo.Name == "unknown" {
		pkgInfo.Name = node.Name.Name
	}

	// Map to track method receivers
	structMethods := make(map[string][]FunctionInfo)

	// Extract imports
	for _, imp := range node.Imports {
		impPath := strings.Trim(imp.Path.Value, "\"")
		if !contains(pkgInfo.Imports, impPath) {
			pkgInfo.Imports = append(pkgInfo.Imports, impPath)
		}
	}

	// Extract declarations
	for _, decl := range node.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			// Process function declaration
			fInfo := FunctionInfo{
				Name:    d.Name.Name,
				Package: pkgInfo.Name,
				File:    filePath,
				Line:    fset.Position(d.Pos()).Line,
				Context: getNodeContentFromFset(d, fset, filePath),
			}

			// Get docstring
			if d.Doc != nil {
				fInfo.DocString = d.Doc.Text()
			}

			// Check if it's a method
			if d.Recv != nil && len(d.Recv.List) > 0 {
				var receiverType string
				switch t := d.Recv.List[0].Type.(type) {
				case *ast.StarExpr:
					if ident, ok := t.X.(*ast.Ident); ok {
						receiverType = ident.Name
					}
				case *ast.Ident:
					receiverType = t.Name
				}

				fInfo.ReceiverType = receiverType

				// Add to method map for later struct association
				if receiverType != "" {
					structMethods[receiverType] = append(structMethods[receiverType], fInfo)
				}
			}

			// Extract function calls with improved detection
			if d.Body != nil {
				ast.Inspect(d.Body, func(n ast.Node) bool {
					if call, ok := n.(*ast.CallExpr); ok {
						switch fun := call.Fun.(type) {
						case *ast.SelectorExpr:
							// Handle method calls (e.g., obj.Method())
							fInfo.Calls = append(fInfo.Calls, fun.Sel.Name)
						case *ast.Ident:
							// Handle direct function calls (e.g., Function())
							fInfo.Calls = append(fInfo.Calls, fun.Name)
						case *ast.FuncLit:
							// Handle anonymous function calls - just add a placeholder
							fInfo.Calls = append(fInfo.Calls, "anonymous")
						case *ast.CallExpr:
							// Handle chained calls like a(b()) - focus on the inner call
							// We continue inspecting and will catch the inner call separately
						default:
							// Other types of calls, just mark as unknown
							fmt.Fprintf(os.Stderr, "Unknown call type: %T in %s\n", fun, fInfo.Name)
						}
					}
					return true
				})
			}

			pkgInfo.Functions = append(pkgInfo.Functions, fInfo)

		case *ast.GenDecl:
			// Look for type declarations
			for _, spec := range d.Specs {
				if ts, ok := spec.(*ast.TypeSpec); ok {
					switch t := ts.Type.(type) {
					case *ast.StructType:
						// Process struct
						sInfo := StructInfo{
							Name:    ts.Name.Name,
							Package: pkgInfo.Name,
							File:    filePath,
							Line:    fset.Position(ts.Pos()).Line,
							Context: fmt.Sprintf("type %s %s", ts.Name.Name, getNodeContentFromFset(t, fset, filePath)),
						}

						// Get docstring
						if d.Doc != nil {
							sInfo.DocString = d.Doc.Text()
						}

						// Extract fields
						for _, field := range t.Fields.List {
							if field.Names != nil && len(field.Names) > 0 {
								fInfo := FieldInfo{
									Name: field.Names[0].Name,
									Type: getTypeString(field.Type),
								}

								if field.Tag != nil {
									fInfo.Tag = field.Tag.Value
								}

								sInfo.Fields = append(sInfo.Fields, fInfo)
							}
						}

						pkgInfo.Structs = append(pkgInfo.Structs, sInfo)

					case *ast.InterfaceType:
						// Process interface
						iInfo := InterfaceInfo{
							Name:    ts.Name.Name,
							Package: pkgInfo.Name,
							File:    filePath,
							Line:    fset.Position(ts.Pos()).Line,
							Context: fmt.Sprintf("type %s %s", ts.Name.Name, getNodeContentFromFset(t, fset, filePath)),
						}

						// Get docstring
						if d.Doc != nil {
							iInfo.DocString = d.Doc.Text()
						}

						// Extract methods
						for _, method := range t.Methods.List {
							if len(method.Names) > 0 {
								iInfo.Methods = append(iInfo.Methods, method.Names[0].Name)
							}
						}

						pkgInfo.Interfaces = append(pkgInfo.Interfaces, iInfo)
					}
				}
			}
		}
	}

	// Associate methods with structs
	for i, structInfo := range pkgInfo.Structs {
		if methods, ok := structMethods[structInfo.Name]; ok {
			pkgInfo.Structs[i].Methods = methods
		}
	}
}

// Get string representation of a type
func getTypeString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + getTypeString(t.X)
	case *ast.ArrayType:
		if t.Len == nil {
			return "[]" + getTypeString(t.Elt)
		}
		return "[?]" + getTypeString(t.Elt)
	case *ast.MapType:
		return "map[" + getTypeString(t.Key) + "]" + getTypeString(t.Value)
	default:
		return "unknown"
	}
}

// Get content of a node using a fileset
func getNodeContentFromFset(node ast.Node, fset *token.FileSet, filename string) string {
	start := fset.Position(node.Pos())
	end := fset.Position(node.End())

	// Read the source file
	content, err := os.ReadFile(filename)
	if err != nil {
		return ""
	}

	// Convert byte positions to array indices
	startOffset := start.Offset
	endOffset := end.Offset

	if startOffset >= 0 && endOffset >= 0 && endOffset <= len(content) {
		return string(content[startOffset:endOffset])
	}
	return ""
}

func getNodeContent(node ast.Node, pkg *packages.Package) string {
	fset := pkg.Fset
	start := fset.Position(node.Pos())
	end := fset.Position(node.End())

	// Read the source file
	content, err := os.ReadFile(start.Filename)
	if err != nil {
		return ""
	}

	// Convert byte positions to array indices
	startOffset := start.Offset
	endOffset := end.Offset

	if startOffset >= 0 && endOffset >= 0 && endOffset <= len(content) {
		return string(content[startOffset:endOffset])
	}
	return ""
}

func analyzeFuncDecl(node *ast.FuncDecl, pkg *packages.Package, file *ast.File) FunctionInfo {
	fset := pkg.Fset
	pos := fset.Position(node.Pos())

	info := FunctionInfo{
		Name:    node.Name.Name,
		Package: pkg.PkgPath,
		File:    pos.Filename,
		Line:    pos.Line,
		Context: getNodeContent(node, pkg),
	}

	// Get receiver type if method
	if node.Recv != nil && len(node.Recv.List) > 0 {
		switch t := node.Recv.List[0].Type.(type) {
		case *ast.StarExpr:
			if ident, ok := t.X.(*ast.Ident); ok {
				info.ReceiverType = ident.Name
			}
		case *ast.Ident:
			info.ReceiverType = t.Name
		}
	}

	// Get docstring
	if node.Doc != nil {
		info.DocString = node.Doc.Text()
	}

	// Get function calls - improved detection
	if node.Body != nil {
		ast.Inspect(node.Body, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				switch fun := call.Fun.(type) {
				case *ast.SelectorExpr:
					// Handle method calls (e.g., obj.Method())
					info.Calls = append(info.Calls, fun.Sel.Name)
				case *ast.Ident:
					// Handle direct function calls (e.g., Function())
					info.Calls = append(info.Calls, fun.Name)
				case *ast.FuncLit:
					// Handle anonymous function calls - just add a placeholder
					info.Calls = append(info.Calls, "anonymous")
				case *ast.CallExpr:
					// Handle chained calls like a(b()) - focus on the inner call
					// We continue inspecting and will catch the inner call separately
				default:
					// Other types of calls, just mark as unknown
					fmt.Fprintf(os.Stderr, "Unknown call type: %T in %s\n", fun, info.Name)
				}
			}
			return true
		})
	}

	return info
}

func analyzeStructType(node *ast.TypeSpec, structType *ast.StructType, pkg *packages.Package, file *ast.File) StructInfo {
	fset := pkg.Fset
	pos := fset.Position(node.Pos())

	info := StructInfo{
		Name:    node.Name.Name,
		Package: pkg.PkgPath,
		File:    pos.Filename,
		Line:    pos.Line,
		Context: fmt.Sprintf("type %s %s", node.Name.Name, getNodeContent(structType, pkg)),
	}

	if node.Doc != nil {
		info.DocString = node.Doc.Text()
	}

	// Get fields
	for _, field := range structType.Fields.List {
		fInfo := FieldInfo{
			Type: types.ExprString(field.Type),
		}

		if len(field.Names) > 0 {
			fInfo.Name = field.Names[0].Name
		}

		if field.Tag != nil {
			fInfo.Tag = field.Tag.Value
		}

		info.Fields = append(info.Fields, fInfo)
	}

	return info
}

func analyzeInterfaceType(node *ast.TypeSpec, interfaceType *ast.InterfaceType, pkg *packages.Package, file *ast.File) InterfaceInfo {
	fset := pkg.Fset
	pos := fset.Position(node.Pos())

	info := InterfaceInfo{
		Name:    node.Name.Name,
		Package: pkg.PkgPath,
		File:    pos.Filename,
		Line:    pos.Line,
		Context: fmt.Sprintf("type %s %s", node.Name.Name, getNodeContent(interfaceType, pkg)),
	}

	if node.Doc != nil {
		info.DocString = node.Doc.Text()
	}

	// Get methods
	for _, method := range interfaceType.Methods.List {
		if len(method.Names) > 0 {
			info.Methods = append(info.Methods, method.Names[0].Name)
		}
	}

	return info
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
