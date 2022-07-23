//
//  VideoCameraType.swift
//
//  Created by Jayce on 22/017/2022.
//  Copyright © 2022 Jayce. All rights reserved.
//

import AVFoundation


enum CameraType : Int {
    case back
    case front
    
    func captureDevice() -> AVCaptureDevice {
        switch self {
        case .front:
            let devices = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .front).devices
            print("devices:\(devices)")
            for device in devices where device.position == .front {
                return device
            }
        default:
            break
        }
        return AVCaptureDevice.default(for: .video)!
    }
}
