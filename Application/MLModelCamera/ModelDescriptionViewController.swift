//
//  ModelDescriptionViewController.swift
//  MLModelCamera
//
//  Created by Jayce on 22/017/2022.
//  Copyright © 2022 Jayce. All rights reserved.
//

import UIKit
import CoreML

class ModelDescriptionViewController: UIViewController {

    var modelDescription: MLModelDescription!

    @IBOutlet weak var textView: UITextView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)

        var metadataText = ""
        modelDescription.metadata.forEach {
            metadataText += "\($0.0.rawValue):\n\($0.1)\n\n"
        }
        
        let text = """
        Input:
        \(modelDescription.inputDescriptionsByName)
        
        Output:
        \(modelDescription.outputDescriptionsByName)
        
        \(metadataText)
        """

        textView.text = text
        textView.contentOffset = .zero
    }
}

