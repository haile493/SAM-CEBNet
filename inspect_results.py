import json

def main():
    with open('outputs/fair_comparison_2026/fair_comparison_2026_summary.json') as f:
        dfair = json.load(f)

    for setting, methods in dfair.items():
        print(f"\n==================== {setting} ({len(methods)} methods) ====================")
        for m in sorted(methods, key=lambda x: x.get('mean_iou', 0), reverse=True):
            bf1 = f"{m.get('boundary_f1', 0)*100:.2f}%" if m.get('boundary_f1') is not None else "N/A"
            iou = f"{m.get('mean_iou', 0)*100:.2f}%"
            dice = f"{m.get('mean_dice', 0)*100:.2f}%"
            p = f"{m.get('precision', 0)*100:.2f}%"
            r = f"{m.get('recall', 0)*100:.2f}%"
            oa = f"{m.get('overall_accuracy', 0)*100:.2f}%"
            name = m.get('name', m.get('method'))
            params = f"{m.get('params_m', 0):.2f}M"
            fps = f"{m.get('fps', 0):.1f}"
            lat = f"{m.get('latency_ms_per_tile', 0):.2f}ms"
            print(f"{name:<15} | IoU: {iou:<7} | Dice: {dice:<7} | P: {p:<7} | R: {r:<7} | OA: {oa:<7} | B-F1: {bf1:<7} | Params: {params:<8} | FPS: {fps:<6} | Lat: {lat}")

if __name__ == "__main__":
    main()
