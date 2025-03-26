package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/types"
	"os"
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
}

type StructInfo struct {
	Name      string      `json:"name"`
	Package   string      `json:"package"`
	File      string      `json:"file"`
	Line      int         `json:"line"`
	DocString string      `json:"docstring,omitempty"`
	Fields    []FieldInfo `json:"fields"`
	Methods   []string    `json:"methods"`
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
	outputFile := flag.String("output", "", "output JSON file")
	flag.Parse()

	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedSyntax |
			packages.NeedTypes | packages.NeedTypesInfo | packages.NeedDeps,
		Tests: false,
	}

	pkgs, err := packages.Load(cfg, *pkgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading package: %v\n", err)
		os.Exit(1)
	}

	result := make(map[string]PackageInfo)

	for _, pkg := range pkgs {
		if pkg.Errors != nil {
			continue
		}

		pkgInfo := PackageInfo{
			Name:  pkg.Name,
			Path:  pkg.PkgPath,
			Files: pkg.GoFiles,
		}

		// Get imports
		for _, f := range pkg.Syntax {
			for _, imp := range f.Imports {
				impPath := strings.Trim(imp.Path.Value, "\"")
				if !contains(pkgInfo.Imports, impPath) {
					pkgInfo.Imports = append(pkgInfo.Imports, impPath)
				}
			}
		}

		// Analyze package contents
		for _, f := range pkg.Syntax {
			ast.Inspect(f, func(n ast.Node) bool {
				switch node := n.(type) {
				case *ast.FuncDecl:
					// Get function info
					fInfo := analyzeFuncDecl(node, pkg, f)
					pkgInfo.Functions = append(pkgInfo.Functions, fInfo)

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

		result[pkg.PkgPath] = pkgInfo
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

func analyzeFuncDecl(node *ast.FuncDecl, pkg *packages.Package, file *ast.File) FunctionInfo {
	fset := pkg.Fset
	pos := fset.Position(node.Pos())

	info := FunctionInfo{
		Name:    node.Name.Name,
		Package: pkg.PkgPath,
		File:    pos.Filename,
		Line:    pos.Line,
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

	// Get function calls
	ast.Inspect(node.Body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				info.Calls = append(info.Calls, sel.Sel.Name)
			} else if ident, ok := call.Fun.(*ast.Ident); ok {
				info.Calls = append(info.Calls, ident.Name)
			}
		}
		return true
	})

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
