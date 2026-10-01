import json

from vollebak_gimbal.drivers.waveshare import WaveshareSerialGimbal


def test_waveshare_move_encoding_is_newline_delimited_json():
    payload = WaveshareSerialGimbal.encode_move(12.345, -6.789, 60, 20)
    assert payload.endswith(b"\n")
    assert json.loads(payload) == {"T": 133, "X": 12.35, "Y": -6.79, "SPD": 60, "ACC": 20}
