; Inno Setup Script for Vestim
; This creates a professional Windows installer

#define MyAppName "Vestim"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Biswanath Dehury"
#define MyAppURL "https://github.com/yourusername/vestim"
#define MyAppExeName "Vestim.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
AppId={{8A8F6C8B-7B5C-4D8E-9F2A-1E3B4C5D6E7F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE
InfoBeforeFile=INSTALL_INFO.txt
OutputDir=installer_output
OutputBaseFilename=vestim-installer-{#MyAppVersion}
SetupIconFile=vestim\gui\resources\icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "serversetup"; Description: "Run server setup wizard after installation"; GroupDescription: "Server Configuration"; Flags: checked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
Source: "dist\Vestim.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "vestim\gui\resources\*"; DestDir: "{app}\resources"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "vestim\remote\*"; DestDir: "{app}\vestim\remote"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "vestim_server_client.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "vestim_universal_client.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "vestim_server_launcher.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "vestim_server_client.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "vestim_server_setup.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup_remote.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "VESTIM_SERVER_CLIENT.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "REMOTE_SETUP_GUIDE.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "DEPLOYMENT_OPTIONS.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "WORKFLOW_EXAMPLE.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup_server.sh"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "hyperparams.json"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{#MyAppName} Server"; Filename: "{app}\vestim_server_client.bat"; WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"
Name: "{group}\{#MyAppName} Universal"; Filename: "python"; Parameters: """{app}\vestim_universal_client.py"""; WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{autodesktop}\{#MyAppName} Server"; Filename: "{app}\vestim_server_client.bat"; WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"; Tasks: desktopicon
Name: "{autodesktop}\{#MyAppName} Universal"; Filename: "python"; Parameters: """{app}\vestim_universal_client.py"""; WorkingDir: "{app}"; IconFilename: "{app}\resources\icon.ico"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
Filename: "python"; Parameters: """{app}\vestim_server_setup.py"""; Description: "Configure Vestim Server connection"; Flags: postinstall; Tasks: serversetup

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
function GetUninstallString(): String;
var
  sUnInstPath: String;
  sUnInstallString: String;
begin
  sUnInstPath := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\{#emit SetupSetting("AppId")}_is1');
  sUnInstallString := '';
  if not RegQueryStringValue(HKLM, sUnInstPath, 'UninstallString', sUnInstallString) then
    RegQueryStringValue(HKCU, sUnInstPath, 'UninstallString', sUnInstallString);
  Result := sUnInstallString;
end;

function IsUpgrade(): Boolean;
begin
  Result := (GetUninstallString() <> '');
end;

function UnInstallOldVersion(): Integer;
var
  sUnInstallString: String;
  iResultCode: Integer;
begin
  Result := 0;
  sUnInstallString := GetUninstallString();
  if sUnInstallString <> '' then begin
    sUnInstallString := RemoveQuotes(sUnInstallString);
    if Exec(sUnInstallString, '/SILENT /NORESTART /SUPPRESSMSGBOXES','', SW_HIDE, ewWaitUntilTerminated, iResultCode) then
      Result := 3
    else
      Result := 2;
  end else
    Result := 1;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep=ssInstall) then
  begin
    if (IsUpgrade()) then
    begin
      UnInstallOldVersion();
    end;
  end;
end;
