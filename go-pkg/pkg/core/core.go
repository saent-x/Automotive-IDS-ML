package core

import (
	"fmt"

	"github.com/saent-x/go-ml-pl/pkg/entities"
	"github.com/sjwhitworth/golearn/base"
)

type CleanedDatasetInfo struct {
	FileInfo entities.FileInfo
}

func SplitFeature_Target(filepath string) (*base.DenseInstances, error) {
	fmt.Println("==> ml-algo [Parsing CSV to Instance]")

	cleanDataGrid, err := base.ParseCSVToInstances(filepath, true)
	if err != nil {
		return nil, err
	}

	fmt.Println("==> ml-algo [Parsing Results]")

	return cleanDataGrid, nil
}
