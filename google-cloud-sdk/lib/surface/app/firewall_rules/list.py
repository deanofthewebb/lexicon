# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Surface for listing all firewall rules."""

from googlecloudsdk.api_lib.app.api import appengine_firewall_api_client as api_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import firewall_rules_util


class List(base.ListCommand):
  """Lists the firewall rules."""

  detailed_help = {
      'DESCRIPTION':
          '{description}',
      'EXAMPLES':
          """\
          To list all App Engine firewall rules, run:

              $ {command}
          """,
  }

  @staticmethod
  def Args(parser):
    parser.display_info.AddFormat(firewall_rules_util.LIST_FORMAT)

  def Run(self, args):
    client = api_client.AppengineFirewallApiClient.GetApiClient('v1beta')
    return client.List()
