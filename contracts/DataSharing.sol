pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract DataSharing is Ownable {
    struct SmartGrid {
        address gridAddress;
        string gridName;
        bool isRegistered;
        // Add more fields as needed (e.g., location, data hash, etc.)
    }

    mapping(address => SmartGrid) public smartGrids;
    mapping(address => mapping(address => bool)) public permissions;

    event SmartGridRegistered(address indexed gridAddress, string gridName);
    event PermissionGranted(address indexed from, address indexed to);
    event PermissionRevoked(address indexed from, address indexed to);
    event ModelUpdated(address indexed updater, string modelHash); // New event

    function registerSmartGrid(string memory _gridName) public {
        require(!smartGrids[msg.sender].isRegistered, "Already registered.");
        smartGrids[msg.sender] = SmartGrid(msg.sender, _gridName, true);
        emit SmartGridRegistered(msg.sender, _gridName);
    }

    // ... (Other functions: grantPermission, revokePermission, checkPermission)

    function updateModel(string memory _modelHash) public { 
        // You might want to add access control here (e.g., only owner or permitted addresses)
        emit ModelUpdated(msg.sender, _modelHash);
    }
}