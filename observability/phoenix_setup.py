"""Launch and configure Arize Phoenix for LRP solver tracing.

If ``arize-phoenix`` is not installed, :func:`setup_phoenix` returns a no-op
tracer so that the rest of the codebase continues to work without tracing.
"""

from __future__ import annotations


def setup_phoenix(project_name: str = "lrp-aiqa-benchmark"):
    """Launch Phoenix locally and register the OpenTelemetry tracer.

    Args:
        project_name: The Phoenix project name shown in the UI.

    Returns:
        An OpenTelemetry ``Tracer`` instance.  If ``arize-phoenix`` is not
        installed, a no-op tracer is returned so callers need not handle the
        absence of Phoenix specially.

    After calling this function, open http://localhost:6006 to view traces.
    """
    try:
        import phoenix as px
        from opentelemetry import trace
        from phoenix.otel import register

        px.launch_app()
        register(project_name=project_name)
        tracer = trace.get_tracer("lrp-solver")
        print(f"[Phoenix] UI available at http://localhost:6006  (project: {project_name})")
        return tracer

    except ImportError as exc:
        print(f"[Phoenix] arize-phoenix not available ({exc}) — using no-op tracer.")
        try:
            from opentelemetry import trace
            return trace.get_tracer("lrp-solver-noop")
        except ImportError:
            return None
