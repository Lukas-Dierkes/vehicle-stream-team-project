from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File

sharepoint_base_url = "https://stuma365.sharepoint.com/sites/Testing_sharepoint/"
sharepoint_user = ""  # your e-mail
sharepoint_password = ""  # your password
folder_in_sharepoint = "Freigegebene%20Dokumente/"


auth = AuthenticationContext(sharepoint_base_url)
auth.acquire_token_for_user(sharepoint_user, sharepoint_password)
ctx = ClientContext(sharepoint_base_url, auth)
web = ctx.web
ctx.load(web)
ctx.execute_query()
print("Connected to SharePoint: ", web.properties["Title"])


def folder_details(ctx, folder_in_sharepoint):
    folder = ctx.web.get_folder_by_server_relative_url(folder_in_sharepoint)
    fold_names = []
    sub_folders = folder.files
    ctx.load(sub_folders)
    ctx.execute_query()
    for s_folder in sub_folders:
        fold_names.append(s_folder.properties["Name"])
    return fold_names


file_list = folder_details(ctx, folder_in_sharepoint)
print(file_list)

# Reading File from SharePoint Folder
sharepoint_file = "/sites/Testing_sharepoint/Freigegebene%20Dokumente/test.xlsx"
file_response = File.open_binary(ctx, sharepoint_file)


print(file_response)


# Saving file to localwith
with open("test.xlsx", "wb") as output_file:
    output_file.write(file_response.content)
