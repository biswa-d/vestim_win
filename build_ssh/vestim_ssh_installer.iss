[Setup]
AppName=Vestim SSH Client
AppVersion=1.0
AppId={{B8C8A1E2-8F9D-4A5B-9C6E-7F8A9B0C1D2E}
AppPublisher=Vestim Team
AppPublisherURL=https://github.com/vestim
AppSupportURL=https://github.com/vestim/support
AppUpdatesURL=https://github.com/vestim/updates
DefaultDirName={autopf}\VestimSSH
DefaultGroupName=Vestim SSH Client
AllowNoIcons=yes
LicenseFile=LICENSE
InfoBeforeFile=VESTIM_SERVER_CLIENT.md
OutputDir=.
OutputBaseFilename=VestimSSHInstaller
SetupIconFile=vestim\gui\resources\icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "ssh_dist\VestimUniversalClient.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "ssh_dist\VestimServerClient.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "ssh_dist\VestimServerSetup.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "ssh_dist\Launch_Vestim.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "ssh_dist\*.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "ssh_dist\setup_server.sh"; DestDir: "{app}"; Flags: ignoreversion; DestName: "setup_server.sh"

[Icons]
Name: "{group}\Vestim Universal Client"; Filename: "{app}\VestimUniversalClient.exe"
Name: "{group}\Vestim Server Client"; Filename: "{app}\VestimServerClient.exe" 
Name: "{group}\Vestim Server Setup"; Filename: "{app}\VestimServerSetup.exe"
Name: "{group}\Launch Vestim"; Filename: "{app}\Launch_Vestim.bat"
Name: "{group}\{cm:UninstallProgram,Vestim SSH Client}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Vestim Universal"; Filename: "{app}\VestimUniversalClient.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VestimUniversalClient.exe"; Description: "{cm:LaunchProgram,Vestim Universal Client}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
