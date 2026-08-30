import { NextResponse } from "next/server";
import { apiBaseUrl } from "@/app/lib/apiBase";
import type { ModelInfo } from "@/app/lib/types";

export async function GET() {
  const res = await fetch(`${apiBaseUrl()}/model/info`, { cache: "no-store" });

  if (!res.ok) {
    const detail = await res.text();
    return NextResponse.json(
      { error: "upstream /model/info failed", status: res.status, detail },
      { status: 502 },
    );
  }

  const body: ModelInfo = await res.json();
  return NextResponse.json(body);
}
