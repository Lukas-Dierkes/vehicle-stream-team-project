from vehicle_stream_pipeline.hello_world import hello_world


def test_example():
    result = hello_world()
    assert result == "Hello World"
