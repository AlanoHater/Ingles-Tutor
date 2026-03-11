/**
 * API Route: /api/chat
 * Proxy al backend FastAPI. Reenvía streaming SSE.
 */

export async function POST(request: Request) {
  const body = await request.json();

  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";

  try {
    const response = await fetch(`${backendUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: body.messages,
        temperature: body.temperature ?? 0.7,
        max_tokens: body.max_tokens ?? 512,
        stream: true,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      return new Response(error, {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Reenviar el stream SSE del backend al frontend
    return new Response(response.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("Error connecting to backend:", error);
    return new Response(
      JSON.stringify({
        error: "No se pudo conectar con el backend. ¿Está corriendo?",
      }),
      {
        status: 502,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
