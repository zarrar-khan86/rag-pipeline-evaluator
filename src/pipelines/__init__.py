from src.pipelines import crag, graph_rag, hyde, rag_fusion

PIPELINES = {
	"rag_fusion": rag_fusion.run,
	"hyde": hyde.run,
	"crag": crag.run,
	"graph_rag": graph_rag.run,
}
